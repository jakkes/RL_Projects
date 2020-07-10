from typing import Callable, Dict

import torch

class MuZeroConfig:
    def __init__(self,
        representation_net_gen: Callable[[], torch.nn.Module]=None,
        prediction_net_gen: Callable[[], torch.nn.Module]=None,
        dynamics_net_gen: Callable[[], torch.nn.Module]=None
    ):
    
        representation_net_gen: Callable[[], torch.nn.Module] = representation_net_gen
        prediction_net_gen: Callable[[], torch.nn.Module] = prediction_net_gen
        dynamics_net_gen: Callable[[], torch.nn.Module] = dynamics_net_gen

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig):
        self.config: MuZeroConfig = config