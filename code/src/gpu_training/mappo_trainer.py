"""Custom Multi-Agent PPO (MAPPO) trainer: parameter/optimizer initialization,
GAE, decentralized-actor/centralized-critic rollout collection (via
jax.lax.scan, no replay buffer), and the PPO update step."""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from mappo_config import MAPPOConfig
from mappo_models import (
    Actor,
    CentralizedCritic,
    PPOTrainState,
    actor_forward,
    critic_forward,
    gaussian_log_prob,
)
from vec_env_utils import vec_reset, vec_step


def create_train_state(rng, config: MAPPOConfig, per_agent_obs_dim, per_env_obs_dim, action_size_per_drone):
    """Initialize actor/critic parameters, optimizers, and the PPOTrainState.
    per_agent_obs_dim is what each drone's (decentralized) actor sees;
    per_env_obs_dim is the full, concatenated observation the centralized
    critic sees."""
    rng1, rng2 = jax.random.split(rng)

    actor_module = Actor(hidden_sizes=tuple(config.policy_hidden), action_dim=action_size_per_drone)
    critic_module = CentralizedCritic(hidden_sizes=tuple(config.vf_hidden))

    dummy_agent_obs = jnp.zeros((1, per_agent_obs_dim))
    dummy_global_obs = jnp.zeros((1, per_env_obs_dim))

    actor_params = actor_module.init(rng1, dummy_agent_obs)
    critic_params = critic_module.init(rng2, dummy_global_obs)

    lr_schedule = optax.linear_schedule(
        init_value=config.lr,
        end_value=config.lr_final,
        transition_steps=config.num_updates,
    )

    policy_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule),
    )
    value_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule),
    )

    policy_state = TrainState.create(apply_fn=actor_module.apply, params=actor_params, tx=policy_tx)
    value_state = TrainState.create(apply_fn=critic_module.apply, params=critic_params, tx=value_tx)
    env_steps = jnp.zeros((config.num_envs,), dtype=jnp.int32)
    ep_returns = jnp.zeros((config.num_envs,), dtype=jnp.float32)

    train_state = PPOTrainState(
        policy_state=policy_state,
        value_state=value_state,
        env_steps=env_steps,
        ep_returns=ep_returns,
    )
    return train_state, actor_module, critic_module


def train(
    env,
    config: MAPPOConfig,
    per_env_obs_dim,
    per_agent_obs_dim,
    action_dim,
    action_size_per_drone,
    num_updates=None,
    progress_fn=lambda *args: None,
):
    """Run the MAPPO training loop and return the final PPOTrainState.
    per_env_obs_dim/per_agent_obs_dim/action_dim come from
    `vec_env_utils.make_env_and_infer(...)`; num_updates overrides
    config.num_updates if given (e.g. for a short smoke test)."""
    num_updates = config.num_updates if num_updates is None else num_updates

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    # Create parameters + optimizer state
    train_state, actor_module, critic_module = create_train_state(
        init_rng, config, per_agent_obs_dim, per_env_obs_dim, action_size_per_drone
    )

    # Initialize vectorized environments
    rng, *sub = jax.random.split(rng, config.num_envs + 1)
    env_states = vec_reset(env, jnp.stack(sub))

    # Local aliases (closed over below) so the nested functions read like the original notebook cell
    num_envs = config.num_envs
    num_drones = config.num_drones
    rollout_steps = config.rollout_steps
    episode_length = config.episode_length
    gamma = config.gamma
    gae_lambda = config.gae_lambda
    clip_eps = config.clip_eps
    entropy_coef = config.entropy_coef
    vf_coef = config.vf_coef
    mini_batch_size = config.mini_batch_size
    num_epochs = config.num_epochs

    # --------------------------------------------------------------
    # 1. GAE
    # --------------------------------------------------------------
    @jax.jit
    def gae_advantages(rewards, values, dones, gamma=gamma, lam=gae_lambda):
        """
        rewards: (T, N)
        values:  (T+1, N) --> Extra last value for bootstrap
        dones:   (T, N)   --> 0/1 floats or bools

        returns:
          advantages: (T, N)
          returns:    (T, N)  == advantages + values[:-1]
        """
        # Reverse sequences for backward scan
        rewards_r = rewards[::-1]
        values_t_r = values[:-1][::-1]   # values[t] reversed
        values_tp1_r = values[1:][::-1]  # values[t+1] reversed
        dones_r = dones[::-1]

        init = jnp.zeros_like(values[0])  # shape (N,)

        def scan_fn(carry, elems):
            r, v, vnext, done = elems
            nonterminal = 1.0 - done
            delta = r + gamma * vnext * nonterminal - v
            adv = delta + gamma * lam * nonterminal * carry
            return adv, adv

        _, advs_rev = jax.lax.scan(
            scan_fn,
            init,
            (rewards_r, values_t_r, values_tp1_r, dones_r),
        )

        advantages = advs_rev[::-1]
        returns = advantages + values[:-1]

        return advantages, returns

    # --------------------------------------------------------------
    # 2. Rollout step: Using jax.lax.scan & carry, no rollout buffer
    # --------------------------------------------------------------
    def rollout_step(carry, _):
        env_states, train_state, rng = carry
        rng, step_key, reset_key = jax.random.split(rng, 3)

        obs = env_states.obs
        per_agent_obs = obs.reshape((num_envs, num_drones, per_agent_obs_dim))

        mean, log_std = actor_forward(
            train_state.policy_state.params,
            train_state.policy_state.apply_fn,
            per_agent_obs,
        )
        eps = jax.random.normal(step_key, mean.shape)
        actions_per_agent = mean + jnp.exp(log_std) * eps
        log_prob_per_agent = gaussian_log_prob(mean, log_std, actions_per_agent)
        log_prob_per_env = jnp.sum(log_prob_per_agent, axis=1)

        values = critic_forward(
            train_state.value_state.params,
            train_state.value_state.apply_fn,
            obs,
        )

        flat_actions = actions_per_agent.reshape((num_envs, -1))
        next_env_states = vec_step(env, env_states, flat_actions)

        env_steps = train_state.env_steps + 1
        truncated = env_steps >= episode_length
        done = jnp.logical_or(next_env_states.done.reshape(num_envs), truncated)

        reset_keys = jax.random.split(reset_key, num_envs)
        reset_states = vec_reset(env, reset_keys)

        def apply_reset(new, reset):
            """Conditionally go to next step or reset based on `done`, broadcasting to match leaf shape."""
            leaf_ndim = new.ndim
            cond = done.reshape((num_envs,) + (1,) * (leaf_ndim - 1))
            return jnp.where(cond, reset, new)

        final_env_states = jax.tree_util.tree_map(apply_reset, next_env_states, reset_states)

        env_steps = jnp.where(done, 0, env_steps)
        ep_returns = train_state.ep_returns + next_env_states.reward
        ep_returns = jnp.where(done, 0.0, ep_returns)

        train_state = PPOTrainState(
            policy_state=train_state.policy_state,
            value_state=train_state.value_state,
            env_steps=env_steps,
            ep_returns=ep_returns,
        )

        transition = (
            obs,
            flat_actions,
            log_prob_per_env,
            next_env_states.reward.reshape(num_envs),
            done,
            values,
        )
        return (final_env_states, train_state, rng), transition

    # --------------------------------------------------------------
    # 3. Minibatch update & PPO loss functions
    # --------------------------------------------------------------
    def minibatch_update(train_state, batch):
        mb_obs, mb_actions, mb_old_log_prob, mb_adv, mb_ret = batch
        mb_shape = mb_obs.shape[0]
        mb_per_agent_obs = mb_obs.reshape((mb_shape, num_drones, per_agent_obs_dim))

        def policy_loss_fn(params):
            mean, log_std = actor_forward(params, train_state.policy_state.apply_fn, mb_per_agent_obs)
            mean_flat = mean.reshape((mb_shape * num_drones, action_size_per_drone))
            log_std_flat = log_std.reshape((mb_shape * num_drones, action_size_per_drone))
            act_flat = mb_actions.reshape((mb_shape * num_drones, action_size_per_drone))

            log_prob_flat = gaussian_log_prob(mean_flat, log_std_flat, act_flat)
            log_prob = log_prob_flat.reshape((mb_shape, num_drones)).sum(axis=1)
            ratio = jnp.exp(log_prob - mb_old_log_prob)

            unclipped = ratio * mb_adv
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            objective = -jnp.mean(jnp.minimum(unclipped, clipped))
            entropy = entropy_coef * jnp.mean(jnp.sum(0.5 * (jnp.log(2 * jnp.pi) + 1) + log_std_flat, axis=-1))

            return objective - entropy

        def value_loss_fn(v_params):
            v = critic_forward(v_params, train_state.value_state.apply_fn, mb_obs)
            return jnp.mean((v - mb_ret) ** 2) * vf_coef

        pi_loss, pi_grads = jax.value_and_grad(policy_loss_fn)(train_state.policy_state.params)
        vf_loss, vf_grads = jax.value_and_grad(value_loss_fn)(train_state.value_state.params)

        pi_updates, pi_opt_state = train_state.policy_state.tx.update(pi_grads, train_state.policy_state.opt_state)
        vf_updates, vf_opt_state = train_state.value_state.tx.update(vf_grads, train_state.value_state.opt_state)

        new_train_state = PPOTrainState(
            policy_state=train_state.policy_state.replace(
                params=optax.apply_updates(train_state.policy_state.params, pi_updates),
                opt_state=pi_opt_state,
            ),
            value_state=train_state.value_state.replace(
                params=optax.apply_updates(train_state.value_state.params, vf_updates),
                opt_state=vf_opt_state,
            ),
            env_steps=train_state.env_steps,
            ep_returns=train_state.ep_returns,
        )

        return new_train_state, None

    # --------------------------------------------------------------
    # 4. PPO update (epochs x minibatches) via lax.scan
    # --------------------------------------------------------------
    def ppo_update(train_state, traj, adv, ret, rng):
        T = rollout_steps
        E = num_envs
        B = T * E  # total transitions for this update

        obs = traj[0].reshape((B, per_env_obs_dim))
        acts = traj[1].reshape((B, action_dim))
        old_logp = traj[2].reshape((B,))
        adv = adv.reshape((B,))
        ret = ret.reshape((B,))

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        rng, perm_key = jax.random.split(rng)
        idx = jax.random.permutation(perm_key, B)
        n_minibatches = B // mini_batch_size

        idx = idx[: n_minibatches * mini_batch_size]
        idx = idx.reshape((n_minibatches, mini_batch_size))

        def minibatch_scan_fn(ts, mb_idx):
            batch = (obs[mb_idx], acts[mb_idx], old_logp[mb_idx], adv[mb_idx], ret[mb_idx])
            return minibatch_update(ts, batch)

        def epoch_scan_fn(ts, _):
            ts, _ = jax.lax.scan(minibatch_scan_fn, ts, idx)
            return ts, None

        train_state, _ = jax.lax.scan(epoch_scan_fn, train_state, None, length=num_epochs)

        return train_state, rng

    # --------------------------------------------------------------
    # 5. Full MAPPO training step: rollout + GAE + PPO update
    # --------------------------------------------------------------
    @jax.jit
    def training_step(train_state, env_states, rng):
        (env_states, train_state, rng), traj = jax.lax.scan(
            rollout_step,
            (env_states, train_state, rng),
            None,
            length=rollout_steps,
        )

        obs, acts, logp, rewards, dones, values = traj

        last_values = critic_forward(
            train_state.value_state.params,
            train_state.value_state.apply_fn,
            obs[-1],  # Global obs
        )
        values = jnp.concatenate([values, last_values[None, :]], axis=0)
        adv, ret = gae_advantages(rewards, values, dones)

        train_state, rng = ppo_update(train_state, traj, adv, ret, rng)

        metrics = {"mean_reward": jnp.mean(train_state.ep_returns)}
        return train_state, env_states, rng, metrics

    # --------------------------------------------------------------
    # 6. Run updates
    # --------------------------------------------------------------
    training_step_jit = jax.jit(training_step)
    for update in range(num_updates):
        train_state, env_states, rng, metrics = training_step_jit(train_state, env_states, rng)
        if update % max(1, num_updates // 10) == 0:
            progress_fn(update, metrics)

    return train_state
