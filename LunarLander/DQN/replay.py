from typing import List, Tuple
from random import randint

class Replay:
    def __init__(self, capacity: int):
        capacity = int(capacity)
        self.capacity = capacity
        self.objects = [None] * capacity   # type: List[Tuple]
        self.position = 0
        self.full_loop = False

    def add(self, o: tuple):
        self.objects[self.position] = o
        self.position += 1
        if self.position >= self.capacity:
            if not self.full_loop:
                self.full_loop = True
            self.position = 0

    def get_random(self) -> Tuple:
        if self.full_loop:
            return self.objects[randint(0, self.capacity - 1)]
        else:
            return self.objects[randint(0, self.position - 1)]

    def count(self):
        return self.capacity if self.full_loop else self.position