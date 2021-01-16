from multiprocessing import Pool

import numpy as np
from rl.simulators import Simulator


def _roll_out(args) -> float:
    simulator, state, action_mask = args
    done = False
    reward = 0.0
    reward_flip = -1
    while not done:
        state, action_mask, reward, done, _ = simulator.step(
            state, np.random.choice(np.arange(action_mask.shape[0])[action_mask]))
        reward_flip *= -1
    return reward * reward_flip


class MonteCarlo:
    def __init__(self, simulator: Simulator, trials: int = 1000, processes: int = 8):
        self.simulator = simulator
        self.pool = Pool(processes)
        self.trials = trials

    def act(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        rewards_per_action = np.zeros(action_mask.shape[0])
        rewards_per_action[~action_mask] = -np.inf
        for i in range(action_mask.shape[0]):
            if not action_mask[i]:
                continue

            next_state, next_mask, reward, done, _ = self.simulator.step(
                state, i)
            if done:
                rewards_per_action[i] = reward
                continue

            roll_out_rewards = self.pool.map(
                _roll_out, [(self.simulator, next_state, next_mask) for _ in range(self.trials)])
            roll_out_rewards = np.array(list(roll_out_rewards))
            rewards_per_action[i] = - np.mean(roll_out_rewards)
        return np.argmax(rewards_per_action)

    def close(self):
        self.pool.close()
