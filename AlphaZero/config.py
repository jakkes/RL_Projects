class AlphaZeroConfig:
    def __init__(self):
        self.c = 1.25
        self.T = 1.0
        self.alpha = 0.5
        self.epsilon = 0.25
        self.batch_size = 32