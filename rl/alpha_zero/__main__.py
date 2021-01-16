from argparse import ArgumentParser
from time import sleep
from random import randrange

import numpy as np
from torch import nn, optim, jit
from torch.multiprocessing import Queue

from rl.simulators import TicTacToe, Simulator
from .config import AlphaZeroConfig
from .learner_worker import LearnerWorker
from .self_play_worker import SelfPlayWorker
from .network import Network
from .loggers import LearnerLogger, SelfPlayLogger
from .mcts import mcts
from .node import Node


parser = ArgumentParser()
parser.add_argument("--save-path", type=str, default=None)
parser.add_argument("--play", action="store_true", default=False)


def train(self_play_workers: int, simulator: Simulator, network: nn.Module, optimizer: optim.Optimizer, config: AlphaZeroConfig, save_path: str=None):

    network.share_memory()

    sample_queue = Queue(maxsize=2000)
    episode_logging_queue = Queue(maxsize=2000)
    learner_logging_queue = Queue(maxsize=2000)

    self_play_workers = [SelfPlayWorker(simulator, network, config, sample_queue,
                                        episode_logging_queue=episode_logging_queue) for _ in range(self_play_workers)]
    learner_worker = LearnerWorker(network, optimizer, config, sample_queue, learner_logging_queue=learner_logging_queue, save_path="./models")

    learner_logger = LearnerLogger(learner_logging_queue)
    self_play_logger = SelfPlayLogger(episode_logging_queue)

    learner_logger.start()
    self_play_logger.start()
    learner_worker.start()
    for worker in self_play_workers:
        worker.start()

    while True:
        sleep(10)


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

        state, mask, reward, terminal, _ = simulator.step(state, action)
        if root is not None:
            root = root.children[action]

    simulator.render(state)


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.play:
        net = Network()
        train(4, TicTacToe, net, optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-5), AlphaZeroConfig(), args.save_path)

    else:
        net = jit.load(args.save_path)
        play(TicTacToe, net, AlphaZeroConfig())
