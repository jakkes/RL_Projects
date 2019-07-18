import os

from dotenv import load_dotenv
load_dotenv()

GAME_WIDTH = int(os.getenv("GAME_WIDTH"))
assert(type(GAME_WIDTH) is int)

GAME_HEIGHT = int(os.getenv("GAME_HEIGHT"))
assert(type(GAME_HEIGHT) is int)

TICKS_PER_SECOND = float(os.getenv("TICKS_PER_SECOND"))
assert(type(TICKS_PER_SECOND) is float)

SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT"))
assert(type(SCREEN_HEIGHT) is int)

SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH"))
assert(type(SCREEN_WIDTH) is int)

START_LENGTH = int(os.getenv("START_LENGTH"))
assert(type(START_LENGTH) is int)

HEIGHT_RATIO = SCREEN_HEIGHT / GAME_HEIGHT
WIDTH_RATIO = SCREEN_WIDTH / GAME_WIDTH