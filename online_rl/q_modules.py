from typing import Any, Dict, List
import jax
import equinox as eqx
from equinox import nn
from jax import random
import jax.numpy as jnp
from online_rl.common import Block, final_linear


class QHead(eqx.Module):
    layers: List[eqx.Module]
    value: nn.Linear
    advantage: nn.Linear

    def __init__(self, input_size, hidden_size, output_size, n_layers, key):
        keys = random.split(key, n_layers + 2)

        self.layers = []
        assert n_layers >= 1
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(eqx.filter_vmap(Block(in_size, hidden_size, keys[i])))

        self.value = eqx.filter_vmap(final_linear(keys[-2], hidden_size, 1, scale=0.01))
        self.advantage = eqx.filter_vmap(
            final_linear(keys[-1], hidden_size, output_size, scale=0.01)
        )

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, len(self.layers) * T)
        for i in range(len(self.layers)):
            x = self.layers[i](x, net_keys[i * T : (i + 1) * T])

        V = self.value(x)
        A = self.advantage(x)
        # Dueling DQN
        return V + (A - A.mean(axis=-1, keepdims=True))


class RecurrentQNetwork(eqx.Module):
    """The core model used in experiments. Optimizer depends on its field names."""

    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    head: eqx.Module

    def __init__(self, obs_shape, act_shape, memory_module, config, key):
        self.memory = memory_module

        self.config = config
        [self.input_size] = obs_shape
        self.output_size = act_shape

        keys = random.split(key, 2)
        self.pre = eqx.filter_vmap(Block(self.input_size, self.memory.d_model, keys[0]))

        self.head = QHead(
            input_size=self.memory.d_model,
            hidden_size=config["head_d_hidden"],
            output_size=act_shape,
            n_layers=config["head_n_layers"],
            key=keys[1],
        )

    def __call__(self, x, state, start, key):
        if x.ndim == 2:
            """
            Training: a long tape of several episodes
            x: (T, d_input) where T is tape length
            state: L[(1, d_hidden)] complex
            start: (T,) bool
            """
            T = x.shape[0]
            net_keys = random.split(key, T + 1)
            x = self.pre(x, net_keys[:T])  # x: (T, d_model)
            y, _ = self.memory(x=x, state=state, start=start, key=key)
            # y: (T, d_model) real

            q = self.head(y, key=net_keys[-1])  # (T, d_action)
            return q
        else:
            """
            Evaluation: batched rollouts
            x: (B, T=1, d_input)
            state, new_state: L[(B, 1, d_hidden)] complex
            start: (B, T=1) bool
            """
            B, T, _ = x.shape
            pre_key, mem_key, q_key = random.split(key, 3)
            pre_keys = random.split(pre_key, B * T)
            mem_keys = random.split(mem_key, B)

            x_flat = x.reshape(B * T, -1)
            x_flat = self.pre(x_flat, pre_keys)  # (B*T, d_model)
            x_bt = x_flat.reshape(B, T, -1)  # (B, T, d_model)

            # vmap on the batch-axis of all leaves in the inputs
            y_bt, new_state = jax.vmap(self.memory)(x_bt, state, start, mem_keys)
            y_bt = y_bt.reshape(B * T, -1)

            q_flat = self.head(y_bt, key=q_key)  # (B*T,d_act)
            q_bt = q_flat.reshape(B, T, -1)  # (B,T,d_act)
            return {"q": q_bt, "new_state": new_state, "feature": y_bt}

    def initial_state(self, shape=tuple()):
        if isinstance(shape, int):
            shape = (shape,)
        return self.memory.initial_state(shape)


class GreedyAgent(eqx.Module):
    q_network: eqx.Module

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return self.q_network.initial_state(shape)

    @eqx.filter_jit
    def __call__(
        self,
        x: jnp.ndarray,
        state: Any,
        start: jnp.ndarray,
        key: random.PRNGKey,
    ):
        output = self.q_network(x, state, start, key=key)
        action = jnp.argmax(output["q"].squeeze(), axis=-1)
        return action, output["new_state"], output["feature"]


def anneal(epsilon_start, epsilon_end, progress):
    return epsilon_start + (epsilon_end - epsilon_start) * progress


class EpsilonGreedyAgent(eqx.Module):
    q_network: eqx.Module
    eps_start: float  # static leaf
    eps_end: float  # static leaf
    progress: jnp.ndarray  # dynamic leaf
    epsilon: jnp.ndarray  # dynamic leaf

    def __init__(self, q_network, eps_start, eps_end, progress):
        self.q_network = q_network
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.progress = progress
        self.epsilon = anneal(self.eps_start, self.eps_end, self.progress)

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return self.q_network.initial_state(shape)

    @eqx.filter_jit
    def __call__(
        self,
        x: jnp.ndarray,
        state: Any,
        start: jnp.ndarray,
        key: random.PRNGKey,
    ):
        p_key, r_key, s_key = random.split(key, 3)
        # greedy action
        output = self.q_network(x, state, start, key=p_key)
        action = jnp.argmax(output["q"].squeeze(), axis=-1)

        random_action = random.randint(
            r_key, action.shape, 0, self.q_network.output_size
        )
        do_random = random.uniform(s_key, action.shape) < self.epsilon
        action = jax.lax.select(
            do_random,
            random_action,
            action,
        )
        return action, output["new_state"], output["feature"]
