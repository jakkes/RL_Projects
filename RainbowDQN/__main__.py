from random import random, randrange

import gym
import torch
from torch.optim import Adam

from . import RainbowAgent, RainbowConfig

if __name__ == "__main__":
    config = RainbowConfig(
        state_dim=8, action_dim=4, device="cpu",
        n_steps=3,
        replay_capacity=4*8192,
        batchsize=32,
        optimizer=Adam, 
        optimizer_params={'lr': 1e-4},
        discount=0.99,
        target_update_freq=10,
        use_distributional=True,
        Vmin=-200,
        Vmax=200,
        no_atoms=51,
        use_double=True,
        use_dueling=True,
        pre_stream_hidden_layer_sizes=[64],
        value_stream_hidden_layer_sizes=[32],
        advantage_stream_hidden_layer_sizes=[32],
        use_noisy=True,
        std_init=0.5,
        use_prioritized_replay=False,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_t_start=0,
        beta_t_end=400
    )
    agent = RainbowAgent(config)
    rewards = []

    env = gym.make("LunarLander-v2")

    for episode in range(10000000):

        done = False
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float)

        tot_reward = 0

        while not done:

            if config.use_noisy or random() > 0.1:
                with torch.no_grad():
                    action = agent.get_actions(state.unsqueeze(0)).item()
            else:
                action = randrange(config.action_dim)
            
            if episode % 10 == 0:
                env.render()
            next_state, reward, done, _ = env.step(action)
            tot_reward += reward
            next_state = torch.as_tensor(next_state, dtype=torch.float)

            agent.observe(state, action, reward, not done, next_state)

            if agent.replay.get_size() > 500:
                agent.train_step()

            state = next_state

        rewards.append(tot_reward)

        if episode % 10 == 0:
            print("""
            Episode - {}
            Reward - {}
            """.format(
                episode, sum(rewards) / 10
            ))
            rewards = []



