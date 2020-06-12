import gym
import torch
from torch.optim import Adam

from . import RainbowAgent, RainbowConfig

if __name__ == "__main__":
    config = RainbowConfig(
        state_dim=4, action_dim=2, device="cpu",
        pre_stream_hidden_layer_sizes=[256], 
        value_stream_hidden_layer_sizes=[64],
        advantage_stream_hidden_layer_sizes=[64], 
        no_atoms=51, 
        Vmax=500, 
        Vmin=0, 
        std_init=0.5, 
        optimizer=Adam, 
        optimizer_params={'lr': 5e-4},
        n_steps=1,
        discount=0.99,
        replay_capacity=4*8192,
        batchsize=32,
        beta_start=0.4,
        beta_end=1.0,
        beta_t_start=0,
        beta_t_end=1000,
        alpha=0.6,
        target_update_freq=25
    )
    agent = RainbowAgent(config)


    env = gym.make("CartPole-v0")

    for episode in range(10000000):

        done = False
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float)

        tot_reward = 0

        while not done:

            with torch.no_grad():
                action = agent.get_actions(state.unsqueeze(0)).item()
            
            next_state, reward, done, _ = env.step(action)
            tot_reward += reward
            next_state = torch.as_tensor(next_state, dtype=torch.float)

            agent.observe(state, action, reward, not done, next_state)

            if agent.replay.get_size() > 2000:
                agent.train_step()

            state = next_state

        print(tot_reward)



