from .config import TrainerConfig


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
