import torch

from environments.go import Go

SIZE = 5

def evaluate_action(state, action, prev_state, prev_action, black_to_play, black_captures, white_captures, num_simulations):
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
        q = evaluate_action(state, action, prev_state, prev_action, black_to_play, black_captures, white_captures, num_simulations)
        if q > qastar:
            astar = action

    return astar

    


def run():
    env = Go(SIZE)
    state = env.reset()

    prev_state  = torch.zeros_like(state)
    prev_action = - 1.0

    simenv = Go(SIZE)



if __name__ == "__main__":
    run()
