import numpy as np
import torch
from torch.multiprocessing import Pool

from environments.go import Go
from utils.random import choice

SIZE = 5
SIMS = 3
# pool = Pool(8)

def evaluate_action(data):
    state, action, prev_state, prev_action, black_to_play, black_captures, white_captures = data
    env = Go(SIZE)
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


def evaluate_actions(state, prev_state, prev_action, black_to_play, black_captures, white_captures, num_simulations):
    
    env = Go(SIZE)
    state = env.reset(
        state=state, prev_state=prev_state, 
        prev_action=prev_action, black_to_play=black_to_play,
        black_captures=black_captures, white_captures=white_captures
    )
    actions = env.action_mask().view(-1)
    actions = torch.arange(actions.shape[0])[actions]

    astar = -1
    qastar = -1

    for action in actions:
        action = int(action)
        evals = list(map(evaluate_action, [(state, action, prev_state, prev_action, black_to_play, black_captures, white_captures) for _ in range(num_simulations)]))
        q = np.mean(evals)
        if q > qastar:
            astar = action

    return astar


def get_input_action():
    try:
        return int(input('Action: '))
    except ValueError:
        return get_input_action()


def run():
    env = Go(SIZE)
    state = env.reset()

    prev_state  = torch.zeros_like(state)
    prev_action = - 1.0

    black = True
    done = False

    while not done:
        env.render()
        if black:
            action = get_input_action()
        else:
            action = evaluate_actions(state, prev_state, prev_action, env._next_is_black, env._black_captures, env._white_captures, SIMS)
        state, reward, done = env.step(action)
        black = not black



if __name__ == "__main__":
    run()
