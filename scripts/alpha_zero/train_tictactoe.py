from argparse import ArgumentParser
from time import sleep

from torch import nn, optim
from torch.multiprocessing import Queue

from rl.simulators import TicTacToe, Simulator
from rl.alpha_zero.config import AlphaZeroConfig
from rl.alpha_zero.learner_worker import LearnerWorker
from rl.alpha_zero.self_play_worker import SelfPlayWorker
from rl.alpha_zero.network import Network
from rl.alpha_zero.loggers import LearnerLogger, SelfPlayLogger


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
    learner_worker = LearnerWorker(network, optimizer, config, sample_queue, learner_logging_queue=learner_logging_queue, save_path=save_path)

    learner_logger = LearnerLogger(learner_logging_queue)
    self_play_logger = SelfPlayLogger(episode_logging_queue)

    learner_logger.start()
    self_play_logger.start()
    learner_worker.start()
    for worker in self_play_workers:
        worker.start()

    while True:
        sleep(10)


if __name__ == "__main__":
    args = parser.parse_args()

    net = Network()
    train(4, TicTacToe, net, optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-5), AlphaZeroConfig(), args.save_path)
