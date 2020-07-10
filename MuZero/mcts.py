from typing import Dict

import torch
from torch import Tensor

class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.children: Dict[int, Node] = {}
        self.value_sum: float = 0
        self.visit_count: int = 0
        self.state: Tensor = None
        self.reward: float = None

class Simulation:
    def __init__(self):
        pass