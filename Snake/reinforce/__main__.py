import torch
import torch.cuda as cuda
from torch.optim import Adam
from torch.nn import MSELoss
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from reinforce.network import Policy, Baseline

from snake.game import Game
from snake.frame import Frame

import arcade
import time

L = 0.99    # Discount
C = 10      # Save frequency
R = 1000    # Render frequency
M = 400
E = 100000   # Episodes
policy_LR = 1e-4
baseline_LR = 2e-4

available_actions = [
    Game.Direction.up,
    Game.Direction.down,
    Game.Direction.left,
    Game.Direction.right
]
available_action_indices = [0, 1, 2, 3]

policy = Policy(); policy.to(device)
baseline = Baseline(); baseline.to(device)

policy_opt = Adam(policy.parameters(), lr=policy_LR)
baseline_opt = Adam(baseline.parameters(), lr=baseline_LR)

frame = Frame()

# The agent essentially sees the snake itself, the head and the apple
def get_state(game):
    state = torch.zeros(3, 20, 20, device=device)
    
    state[1, game.player.positions[0].x, game.player.positions[0].y] = 1
    state[2, game.apple.x, game.apple.y] = 1

    for pos in game.player.positions:
        state[0, pos.x, pos.y] = 1

    return state

for e in range(E):
    
    states = []
    action_indices = []
    rewards = []
    action_dists = []

    game = Game()
    frame.game = game
    state = get_state(game)

    tot_reward = 0.0
    initial_value = baseline(state.view(1, 3, 20, 20)).item()
    
    steps_with_same_length = 0

    while True:

        if e % R == 0:
            frame.on_draw()
            arcade.finish_render()
            time.sleep(0.01)

        action_dist = policy(state.view(1, 3, 20, 20)).view(-1)
        action_index = choices(available_action_indices, weights=action_dist, k=1)[0]
        action = available_actions[action_index]

        pre_length = game.player.length
        game.set_direction(action)
        game.tick()
        
        done = game.status == Game.Status.Dead
        if done:
            reward = -1
        else:
            next_state = get_state(game)
            if game.player.length > pre_length:
                reward = 1
                steps_with_same_length = 0
            else:
                reward = 0
                steps_with_same_length += 1
            if steps_with_same_length > M:
                done = True
                reward = -1
        tot_reward += reward

        states.append(state)
        action_indices.append(action_index)
        action_dists.append(action_dist)
        rewards.append(reward)
        
        if done:
            break
        else:
            state = next_state

    n = len(rewards)
    G = torch.empty(n)
    G[-1] = rewards[-1]
    for i in range(2, n+1):
        G[-i] = rewards[-i] + L * G[-(i-1)]

    V = baseline(torch.stack(states)).view(-1)
    delta = torch.pow(L, torch.arange(0, n).float()) * (G - V)
    
    baseline_loss = delta.pow(2).mean()
    policy_loss = - (delta.detach() * torch.log(torch.stack(action_dists)[torch.arange(0,n), action_indices])).mean()

    baseline_opt.zero_grad()
    baseline_loss.backward()
    baseline_opt.step()

    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    print("Episode {} - Reward {} - Discounted reward {} - Initial V {}".format(e, tot_reward, G[0], initial_value))

    if e % C == 0 and e != 0:
        torch.save(policy.state_dict(), './reinforce/models/policy')
        torch.save(baseline.state_dict(), './reinforce/models/baseline')