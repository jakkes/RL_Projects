import threading
import time

import arcade

import snake.env as env
from snake.game import Game

class Frame(arcade.Window):
    def __init__(self):
        super(Frame, self).__init__(env.SCREEN_WIDTH, env.SCREEN_HEIGHT)
        arcade.set_background_color(arcade.color.WHITE)

        self.stopped = False
        self.game_loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.game = Game()

    def _loop(self):
        while self.game.status == Game.Status.Alive and not self.stopped:
            tick_start = time.perf_counter()
            self.game.tick()
            tick_duration = (time.perf_counter() - tick_start) // 1000
            time.sleep(1 / env.TICKS_PER_SECOND - tick_duration)
            
    def on_key_press(self, symbol, modifier):
        if symbol == arcade.key.RIGHT:
            self.game.set_direction(Game.Direction.right)
        elif symbol == arcade.key.LEFT:
            self.game.set_direction(Game.Direction.left)
        elif symbol == arcade.key.UP:
            self.game.set_direction(Game.Direction.up)
        elif symbol == arcade.key.DOWN:
            self.game.set_direction(Game.Direction.down)

    def _draw_player(self):
        for pos in self.game.player.positions:
            arcade.draw_xywh_rectangle_filled(
                env.WIDTH_RATIO * pos.x,
                env.HEIGHT_RATIO * pos.y,
                env.WIDTH_RATIO,
                env.HEIGHT_RATIO,
                arcade.color.GREEN
            )

    def _draw_apple(self):
        arcade.draw_xywh_rectangle_filled(
                env.WIDTH_RATIO * self.game.apple.x,
                env.HEIGHT_RATIO * self.game.apple.y,
                env.WIDTH_RATIO,
                env.HEIGHT_RATIO,
                arcade.color.RED
            )

    def on_draw(self):
        arcade.start_render()

        self._draw_player()
        self._draw_apple()

        # Automatically called since arcade.Window is derived from
        # arcade.finish_render()

    def start(self):
        self.game_loop_thread.start()

    def stop(self):
        self.stopped = True
        self.game_loop_thread.join()
