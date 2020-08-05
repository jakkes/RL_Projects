import gym

import torch
from torch import nn, optim, Tensor

from utils.env import ParallelEnv

from . import PPOAgent, PPOConfig

ACTORS = 8
TRAIN_STEPS = 16 # per actor
EPOCHS = 10
STEPS = 1   # steps to repeat action

if __name__ == "__main__":
    val_net = lambda: nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(inplace=True),
        nn.Linear(32, 32), nn.ReLU(inplace=True),
        nn.Linear(32, 1)
    )

    pol_net = lambda: nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(inplace=True),
        nn.Linear(32, 32), nn.ReLU(inplace=True),
        nn.Linear(32, 2), nn.Softmax(dim=1)
    )

    config = PPOConfig(
        epsilon=0.1,
        policy_net_gen=pol_net,
        value_net_gen=val_net,
        optimizer=optim.Adam,
        optimizer_params={'lr': 1e-4},
        discount=0.99,
        gae_discount=0.95
    )

    agent = PPOAgent(config)
    
    ###############
    ## For stats ##
    mean_reward = 0.0
    start_value = 0.0
    tot_rewards = [0.0] * ACTORS
    ## For stats ##
    ###############

    env_gen_fn = lambda: gym.make("CartPole-v0")
    env = ParallelEnv(env_gen_fn, ACTORS, no_repeats=STEPS)
    start_states = env.reset()
    state_shape = start_states.shape[1:]

    states = torch.empty(ACTORS, TRAIN_STEPS+1, *state_shape)
    actions = torch.empty(ACTORS, TRAIN_STEPS, dtype=torch.long)
    rewards = torch.empty(ACTORS, TRAIN_STEPS)
    not_dones = torch.empty(ACTORS, TRAIN_STEPS, dtype=torch.bool)
    states[:, -1] = start_states

    mean_reward = 0.0
    total_rewards = torch.zeros(ACTORS)
    while True:

        states[:, 0] = states[:, -1]
        for k in range(TRAIN_STEPS):
            
            with torch.no_grad():
                actions[:, k] = agent.get_actions(states[:, k])
            
            s, r, d, _ = env.step(actions[:, k])
            states[:, k+1] = s
            rewards[:, k] = r
            not_dones[:, k] = ~d

            total_rewards += r

            if any(d):
                states[d, k+1] = env.reset(d)
                mean_reward += 0.1 * (total_rewards[d].mean().item() - mean_reward)
                total_rewards[d] = 0.0
                print(mean_reward)
            env.envs[0].render()

        agent.train_step(states, actions, rewards, not_dones, EPOCHS)
