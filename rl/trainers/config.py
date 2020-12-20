from abc import abstractmethod


class TrainerConfig:

    @abstractmethod
    def get_trainer(self):
        raise NotImplementedError
