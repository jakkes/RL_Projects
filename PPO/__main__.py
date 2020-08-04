import gym

import torch
from torch import nn, optim, Tensor

from . import PPOAgent, PPOConfig

ACTORS = 4
TRAIN_STEPS = 4 # per actor
EPOCHS = 5
BATCHSIZE = 8
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
        epsilon=0.2,
        policy_net_gen=pol_net,
        value_net_gen=val_net,
        optimizer=optim.Adam,
        optimizer_params={'lr': 1e-4, 'eps': 1e-4},
        discount=0.99
    )

    agent = PPOAgent(config)

    mean_reward = 0.0
    start_value = 0.0
    envs = [gym.make("CartPole-v0") for _ in range(ACTORS)]
    dones = [False] * ACTORS
    states = torch.stack([torch.as_tensor(env.reset(), dtype=torch.float) for env in envs])
    tot_rewards = [0.0] * ACTORS

    count = 0
    render_count = 0
    while True:
        for i in range(ACTORS):
            if dones[i]:
                states[i] = torch.as_tensor(envs[i].reset(), dtype=torch.float)
                mean_reward += 0.1 * (tot_rewards[i] - mean_reward)
                start_value += 0.1 * (agent.value_net(states[i].view(1, -1)).item() - start_value)
                tot_rewards[i] = 0.0

                if i == 0:
                    print("""
                    Mean reward - {}
                    Start value - {}
                    """.format(
                        mean_reward, start_value
                    ))
                    render_count += 1

        if render_count % 10 == 0:
            envs[0].render()
        actions = agent.get_actions(states)
        for i in range(ACTORS):
            action = actions[i].item(); state = states[i]
            reward = 0.0
            for _ in range(STEPS):
                next_state, r, done, _ = envs[i].step(action)
                reward += r
                if done:
                    break
            next_state = torch.as_tensor(next_state, dtype=torch.float)
            tot_rewards[i] += reward
            agent.observe(state, action, reward, not done, next_state)
            states[i] = next_state
            dones[i] = done

        count += 1
        if count % TRAIN_STEPS  == 0:
            agent.train_step(EPOCHS, BATCHSIZE)
