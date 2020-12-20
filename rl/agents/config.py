from abc import abstractmethod


class AgentConfig:

    @abstractmethod
    def get_agent(self):
        raise NotImplementedError
