from .config import AgentConfig


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config