from random import choice
import snake.env as env
from snake.point import Point

class Game:
    def __init__(self):
        
        self.player = Player(Point.get_random())
        self.direction = choice([
            Game.Direction.left,
            Game.Direction.right,
            Game.Direction.up,
            Game.Direction.down
        ])

        self.apple = Point.get_random()
        self.status = Game.Status.Alive
            
    def set_direction(self, direction: Point):
        
        if len(self.player.positions) <= 1:
            self.direction = direction
            return

        difference_since_last_step = self.player.get_head() - self.player.get_position_idx(1)
        if direction == Game.Direction.right and difference_since_last_step != Game.Direction.left:
            self.direction = Game.Direction.right
        elif direction == Game.Direction.left and difference_since_last_step != Game.Direction.right:
            self.direction = Game.Direction.left
        elif direction == Game.Direction.up and difference_since_last_step != Game.Direction.down:
            self.direction = Game.Direction.up
        elif direction == Game.Direction.down and difference_since_last_step != Game.Direction.up:
            self.direction = Game.Direction.down

    def tick(self):
        self.player.move(self.direction)
        if self.player.check_collision_walls() or self.player.check_self_collision():
            self.status = Game.Status.Dead

        if self.player.get_head() == self.apple:
            self.apple = Point.get_random()
            self.player.grow()


    class Direction:
        left = Point(-1, 0)
        right = Point(1, 0)
        up = Point(0, 1)
        down = Point(0, -1)

    class Status: 
        Alive = 1
        Dead = -1

class Player:
    def __init__(self, start: Point):
        self.positions = [start]
        self.length = env.START_LENGTH

    def move(self, direction: Point):
        self.positions = [self.get_head() + direction] + self.positions[0:self.length - 1]

    def get_head(self):
        return self.get_position_idx(0)

    def get_position_idx(self, idx: int):
        return self.positions[idx]

    def check_collision_walls(self):
        head = self.get_head()
        return head.x >= env.GAME_WIDTH or head.x < 0 or head.y < 0 or head.y >= env.GAME_HEIGHT
            
    def check_self_collision(self):
        head = self.get_head()
        return any(head == self.get_position_idx(i) for i in range(1, len(self.positions)))

    def grow(self):
        self.length += 2