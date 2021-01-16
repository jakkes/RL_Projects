from typing import Any
from torch.multiprocessing import Queue
from .summary_writer_server import SummaryWriterServer


class LearnerLogger(SummaryWriterServer):
    def __init__(self, data_queue: Queue):
        super().__init__("Learner", data_queue)
        self.step = 0

    def log(self, data):
        self.step += 1
        self.summary_writer.add_scalar("Training/Loss", data, global_step=self.step)


class SelfPlayLogger(SummaryWriterServer):
    def __init__(self, data_queue: Queue):
        super().__init__("SelfPlay", data_queue)
        self.step = 0

    def log(self, data: Any):
        self.step += 1
        reward, start_value, kl_div, first_action = data
        self.summary_writer.add_scalar("Episode/Reward", reward, global_step=self.step)
        self.summary_writer.add_scalar("Episode/Start value", start_value, global_step=self.step)
        self.summary_writer.add_scalar("Episode/Start KL Div", kl_div, global_step=self.step)
        self.summary_writer.add_scalar("Episode/First action", first_action, global_step=self.step)