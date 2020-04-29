import torch
from random import randint

class Replay:
    def __init__(self, capacity: int):
        capacity = int(capacity)
        self.capacity = capacity
        
        self.states = torch.zeros(capacity, 8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.not_dones = torch.zeros(capacity, dtype=torch.bool)
        self.next_states = torch.zeros(capacity, 8)

        self.position = 0
        self.full_loop = False

    def add(self, state, action, reward, not_done, next_state):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.not_dones[self.position] = not_done
        self.next_states[self.position] = next_state
        
        self.position += 1
        if self.position >= self.capacity:
            if not self.full_loop:
                self.full_loop = True
            self.position = 0

    def sample(self, n):
        i = torch.randint(self.count(), (n, ))
        return self.states[i], self.actions[i], self.rewards[i], self.not_dones[i], self.next_states[i]

    def count(self):
        return self.capacity if self.full_loop else self.position