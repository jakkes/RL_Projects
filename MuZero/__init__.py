from typing import Callable, Dict

import torch

class MuZeroConfig:
    def __init__(self,
        representation_net_gen: Callable[[], torch.nn.Module]=None,
        prediction_net_gen: Callable[[], torch.nn.Module]=None,
        dynamics_net_gen: Callable[[], torch.nn.Module]=None,
        c1: float=None,
        c2: float=None,
        simulations: int=None,
        discount: float=None,
        policy_temperature: float=None
    ):
    
        self.representation_net_gen: Callable[[], torch.nn.Module] = representation_net_gen
        self.prediction_net_gen: Callable[[], torch.nn.Module] = prediction_net_gen
        self.dynamics_net_gen: Callable[[], torch.nn.Module] = dynamics_net_gen
        self.c1: float = c1
        self.c2: float = c2
        self.simulations: int = simulations
        self.discount: float = discount
        self.policy_temperature: float = policy_temperature

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig):
        self.config: MuZeroConfig = config

        self.representation_net = config.representation_net_gen()
        self.prediction_net = config.prediction_net_gen()
        self.dynamics_net = config.dynamics_net_gen()