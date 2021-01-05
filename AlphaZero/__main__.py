from time import sleep
from typing import Any

from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.multiprocessing import Queue

from simulators import TicTacToe, Simulator
from .config import AlphaZeroConfig
from .learner_worker import LearnerWorker
from .self_play_worker import SelfPlayWorker
from .summary_writer_server import SummaryWriterServer
from .network import Network
from .loggers import LearnerLogger, SelfPlayLogger


def run(self_play_workers: int, simulator: Simulator, network: nn.Module, optimizer: optim.Optimizer, config: AlphaZeroConfig):

    network.share_memory()

    sample_queue = Queue(maxsize=2000)
    episode_logging_queue = Queue(maxsize=2000)
    learner_logging_queue = Queue(maxsize=2000)

    self_play_workers = [SelfPlayWorker(simulator, network, config, sample_queue,
                                        episode_logging_queue=episode_logging_queue) for _ in range(self_play_workers)]
    learner_worker = LearnerWorker(network, optimizer, config, sample_queue, learner_logging_queue=learner_logging_queue)

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
    net = Network()
    run(4, TicTacToe, net, optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4), AlphaZeroConfig())
