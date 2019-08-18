import gym
import matplotlib.pyplot as plt
import torch
import torch.cuda as cuda
from torch.optim import Adam, SGD
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from a2c.network import Actor, Critic

L = 0.99    # Discount
render = True
C = 10      # How often to render the episode.
E = 5000   # Episodes
actor_LR = 1e-4
critic_LR = 1e-3

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    available_actions = [0, 1]

    actor = Actor(); actor.to(device)
    critic = Critic(); critic.to(device)

    actor_opt = SGD(actor.parameters(), lr=actor_LR)
    critic_opt = SGD(critic.parameters(), lr=critic_LR)

    final_rewards = []

    for e in range(E):
        
        state = torch.as_tensor(env.reset(), device=device, dtype=torch.float).view(1, 4)

        tot_reward = 0.0
        initial_value = critic(state).item()
        
        while True:

            action_dist = actor(state).view(-1)
            action = choices(available_actions, weights=action_dist, k=1)[0]

            next_state, reward, done, _ = env.step(action)
            next_state = torch.as_tensor(next_state, device=device, dtype=torch.float).view(1, 4)
            
            if render and e % C == 0:
                env.render()

            tot_reward += reward

            if done:
                td_target = reward
            else:
                td_target = reward + L * critic(next_state).detach()
            V = critic(state)
            delta = td_target - V.detach()

            critic.train(True)
            critic_loss = - (delta * V).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            critic.train(False)

            actor.train(True)
            actor_loss = - (delta * torch.log(action_dist[action])).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            actor.train(False)

            state = next_state
            if done:
                break

        print("Episode {} - Reward {} - Initial value {}".format(
            e, tot_reward, initial_value
        ))

        final_rewards.append(tot_reward)

    avg = torch.as_tensor(final_rewards, dtype=torch.float).cumsum(dim=0) / torch.arange(1, E+1, dtype=torch.float)

    plt.plot(range(1, E+1), final_rewards, label="Episodic reward")
    plt.plot(range(1, E+1), list(avg), label="Average reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()