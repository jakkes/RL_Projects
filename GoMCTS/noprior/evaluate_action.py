import torch

from environments.go import Go
from utils.random import choice

def evaluate_action(data):
    state, action, prev_state, prev_action, black_to_play, black_captures, white_captures = data
    env = Go(state.shape[-1])
    state = env.reset(
        state=state, prev_state=prev_state, 
        prev_action=prev_action, black_to_play=black_to_play,
        black_captures=black_captures, white_captures=white_captures
    )
    flip = 1.0
    done = False
    _, reward, done = env.step(action)

    while not done:
        flip *= -1
        action_mask = env.action_mask()
        prob = torch.ones_like(action_mask).float()
        prob[~action_mask] = 0
        prob /= prob.sum()
        a = int(choice(prob.view(1, -1)))
        _, reward, done = env.step(a)

    return reward * flip