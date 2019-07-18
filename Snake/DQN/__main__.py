from snake.game import Game
from snake.frame import Frame
from DQN.replay import Replay
from DQN.network import Net

import arcade

import torch
from torch.optim import Adam
from torch.nn import MSELoss

import snake.env as env
from typing import List
from snake.point import Point

from copy import deepcopy
from random import randint, random
from time import sleep

C = 1000      # How often to update target network
M = 100000   # Replay memory size
TRAIN_START = 1000
EPS = 1.0  # Epsilon greedy
MIN_EPS = 0.0001
DECAY_EPS = 0.997
Lambda = 0.99   # Discount
B = 64      # Batch size

available_actions = [
    Game.Direction.left,
    Game.Direction.right,
    Game.Direction.up,
    Game.Direction.down
]

replay = Replay(M, (3, 20, 20))
network = Net()
target = deepcopy(network)

opt = Adam(network.parameters(), lr=1e-3)

def get_state(game):
    state = torch.zeros(3, env.GAME_WIDTH, env.GAME_HEIGHT)
    
    state[1, game.player.positions[0].x, game.player.positions[0].y] = 1
    state[2, game.apple.x, game.apple.y] = 1

    for pos in game.player.positions:
        state[0, pos.x, pos.y] = 1

    return state

EPS = EPS / DECAY_EPS**60

frame = Frame()

episode = 0
steps = 0
while True:
    episode += 1
    
    game = Game()
    prev_state = None
    state = get_state(game)

    frame.game = game
    
    print("Episode {} - Steps {} - Epsilon {} - Q-value {} - Memory size {}".format(
        episode, steps, EPS, network(state.view(1, 3, 20, 20)).max().item(), replay.count()
    ))

    while True:

        steps += 1
        if steps % C == 0:
            target = deepcopy(network)
            
        pre_length = game.player.length
        
        if random() < EPS or replay.count() < TRAIN_START:
            action = randint(0, 3)
        else:
            action = network(state.view((1, ) + state.shape)).argmax()
        game.set_direction(available_actions[action])

        game.tick()

        frame.on_draw()
        arcade.finish_render()
        
        prev_state = state
        done = game.status == Game.Status.Dead
        if not done:
            state = get_state(game)
        else:
            state = 0
        reward = 10 if game.player.length > pre_length else 0
        reward = -5 if done else reward

        replay.add(prev_state, state, action, reward, done)

        if replay.count() > TRAIN_START:

            states, next_states, actions, rewards, dones = replay.get_random(B)

            Q = network(states)         # type: torch.Tensor
            Q = Q[torch.arange(0,B), actions]
            
            target_Q = rewards + Lambda * dones * torch.max(target(next_states), dim=1).values

            loss = - ((target_Q.detach() - Q.detach()) * Q).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if done:
            break
            
    EPS = EPS * DECAY_EPS
    if EPS < MIN_EPS:
        EPS = MIN_EPS
