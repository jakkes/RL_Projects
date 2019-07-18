from random import randint
import snake.env as env

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

    @staticmethod
    def get_random(padding=0):
        return Point(randint(0+padding, env.GAME_WIDTH-1-padding), randint(0+padding, env.GAME_HEIGHT-1-padding))