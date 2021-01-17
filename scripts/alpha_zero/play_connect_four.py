from argparse import ArgumentParser
from random import randrange

import numpy as np
from torch import nn, jit

from rl.simulators import ConnectFour, Simulator
from rl.alpha_zero.config import AlphaZeroConfig
from rl.alpha_zero.mcts import mcts
from rl.alpha_zero.node import Node


parser = ArgumentParser()
parser.add_argument("--save-path", type=str, default=None)


def play(simulator: Simulator, network: nn.Module, config: AlphaZeroConfig):
    step = randrange(2)

    state, mask = simulator.reset()
    terminal = False
    root: Node = None

    while not terminal:
        step += 1
        simulator.render(state)

        if step % 2 == 0:
            action = int(input("Action: "))
        else:
            root = mcts(state, mask, simulator,
                        network, config, root_node=root, simulations=config.simulations)
            action = np.random.choice(mask.shape[0], p=root.action_policy)

        state, mask, _, terminal, _ = simulator.step(state, action)
        if root is not None:
            root = root.children[action]

    simulator.render(state)


if __name__ == "__main__":
    args = parser.parse_args()

    net = jit.load(args.save_path)
    play(ConnectFour, net, AlphaZeroConfig())
