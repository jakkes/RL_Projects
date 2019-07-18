import torch
import torch.cuda as cuda
from torch.optim import Adam
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from a2c.network import Actor, Critic

from snake.game import Game
from snake.frame import Frame

import arcade
import time

L = 0.99    # Discount
C = 50      # Save frequency
R = 1000    # Render frequency
E = 10000000   # Episodes
M = 400        # Steps allowed without eating
actor_LR = 1e-4
critic_LR = 5e-4

available_actions = [
    Game.Direction.up,
    Game.Direction.down,
    Game.Direction.left,
    Game.Direction.right
]
available_action_indices = [0, 1, 2, 3]

actor = Actor(); actor.to(device)
actor.load_state_dict(torch.load('./a2c/models/actor', map_location=device))
critic = Critic(); critic.to(device)
critic.load_state_dict(torch.load('./a2c/models/critic', map_location=device))

actor_opt = Adam(actor.parameters(), lr=actor_LR)
critic_opt = Adam(critic.parameters(), lr=critic_LR)

frame = Frame()

# The agent essentially sees the snake itself, the head and the apple
def get_state(game):
    state = torch.zeros(1, 3, 20, 20, device=device)
    
    state[0, 1, game.player.positions[0].x, game.player.positions[0].y] = 1
    state[0, 2, game.apple.x, game.apple.y] = 1

    for pos in game.player.positions:
        state[0, 0, pos.x, pos.y] = 1

    return state

for e in range(E):
    
    game = Game()
    frame.game = game
    state = get_state(game)
    
    tot_reward = 0.0
    initial_value = critic(state).item()
    
    steps_with_same_length = 0

    while True:

        if e % R == 0:
            frame.on_draw()
            arcade.finish_render()
            time.sleep(0.01)

        action_dist = actor(state).view(-1)
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

        if done:
            td_target = reward
        else:
            td_target = reward + L * critic(next_state).detach()
        V = critic(state)
        delta = td_target - V.detach()

        critic_loss = - (delta * V).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_loss = - (delta * torch.log(action_dist[action_index])).mean()     # type: torch.Tensor
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        state = next_state
        if done:
            break

    print("Episode {} - Reward {} - Initial value {}".format(
        e, tot_reward, initial_value
    ))

    if e % C == 0:
        torch.save(actor.state_dict(), './a2c/models/actor')
        torch.save(critic.state_dict(), './a2c/models/critic')