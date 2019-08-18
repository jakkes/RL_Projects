import numpy as np
import arcade

from random import choice
from typing import Tuple
from random import randint

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

    def __eq__(self, point):
        return self.x == point.x and self.y == point.y

    def checker_board_distance_to(self, other) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    @staticmethod
    def get_random(width, height):
        return Point(randint(0, width-1), randint(0, height-1))

class Snake:
    def __init__(self, version: int = 0):
        
        self._version = 0
        self._conf = Config.get_version_config(version) # type: Config

        self._available_actions = [
            Snake.Direction.left,
            Snake.Direction.right,
            Snake.Direction.up,
            Snake.Direction.down
        ]

        self.player = None
        self.apple = None
        self.should_reset = True

        self._frame = None

    def reset(self):
        self.player = Player(Point.get_random(self._conf.width, self._conf.height))
        self.apple = Point.get_random(self._conf.width, self._conf.height)
        self.should_reset = False
        return self._get_state()
    
    def step(self, action: int) -> Tuple:
        if self.should_reset:
            raise ValueError("Must call reset() when done=True has been returned once.")
        if action < 0 or action > 3:
            raise ValueError("Action must be an integer in [0, 3].")
        
        self.player.move(self._available_actions[action])

        done = self._check_collision_walls() or self._check_self_collision()
        if done:
            reward = -10
            self.should_reset = True
        elif self.player.get_head() == self.apple:
            reward = 10
            self.apple = Point.get_random(self._conf.width, self._conf.height)
            self.player.grow()
        else:
            reward = 0.1 if self.player.get_head().checker_board_distance_to(self.apple) < self.player.get_position_idx(1).checker_board_distance_to(self.apple) else -0.1
        state = self._get_state()

        return state, reward, done, {}

    def _check_collision_walls(self):
        head = self.player.get_head()
        return head.x >= self._conf.width or head.x < 0 or head.y < 0 or head.y >= self._conf.height
            
    def _check_self_collision(self):
        head = self.player.get_head()
        return any(head == self.player.get_position_idx(i) for i in range(1, len(self.player.positions)))

    def _get_state(self):
        state = np.zeros((3, self._conf.width, self._conf.height))
        
        for pos in self.player.positions:
            try:
                state[0, pos.x, pos.y] = 1
            except IndexError:
                pass
        state[1, self.apple.x, self.apple.y] = 1
        try:
            state[2, self.player.positions[0].x, self.player.positions[0].y] = 1
        except IndexError:
            pass
        return state

    def render(self):
        if self._frame is None:
            self._frame = arcade.Window(10 * self._conf.width, 10 * self._conf.height)

        arcade.start_render()
        self._draw_player()
        self._draw_apple()
        arcade.finish_render()
    
    def _draw_player(self):
        for pos in self.player.positions:
            arcade.draw_xywh_rectangle_filled(
                10 * pos.x,
                10 * pos.y,
                10,
                10,
                arcade.color.GREEN
            )

    def _draw_apple(self):
        arcade.draw_xywh_rectangle_filled(
                10 * self.apple.x,
                10 * self.apple.y,
                10,
                10,
                arcade.color.RED
            )

    class Direction:
        left = Point(-1, 0)
        right = Point(1, 0)
        up = Point(0, 1)
        down = Point(0, -1)


class Player:
    def __init__(self, start: Point):
        self.positions = [start]
        self.length = 5

    def move(self, direction: Point):
        self.positions = [self.get_head() + direction] + self.positions[0:self.length - 1]

    def get_head(self):
        return self.get_position_idx(0)

    def get_position_idx(self, idx: int):
        return self.positions[idx]

    def grow(self):
        self.length += 1


class Config:

    def __init__(self, width: int, height: int, start_length: int):
        self.width = width
        self.height = height
        self.start_length = start_length

    @staticmethod
    def get_version_config(version: int):
        if version == 0:
            return Config(5, 5, 3)
        elif version == 1:
            return Config(20, 20, 5)
        else:
            raise ValueError("Unknown version {}".format(version))