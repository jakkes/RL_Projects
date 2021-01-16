from abc import abstractmethod
from queue import Empty
from multiprocessing import Process, Queue
from typing import Any
from torch.utils.tensorboard.writer import SummaryWriter


class SummaryWriterServer(Process):
    def __init__(self, filename_suffix: str, data_queue: Queue):
        super().__init__()
        self.data_queue = data_queue
        self.filename_suffix = filename_suffix
        self.summary_writer: SummaryWriter = None

    @abstractmethod
    def log(self, data: Any):
        raise NotImplementedError

    def run(self):

        self.summary_writer = SummaryWriter(comment=self.filename_suffix,)

        while True:
            try:
                data = self.data_queue.get(timeout=5)
            except Empty:
                continue

            self.log(data)
