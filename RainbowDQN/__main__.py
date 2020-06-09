from torch.optim import Adam

from . import RainbowAgent, RainbowConfig

if __name__ == "__main__":
    config = RainbowConfig(state_dim=8, action_dim=4, pre_stream_hidden_layer_sizes=[10, 20, 30], value_stream_hidden_layer_sizes=[30, 20, 10],
    advantage_stream_hidden_layer_sizes=[30, 20, 10], device="cpu", no_atoms=51, Vmax=10, Vmin=-10, std_init=0.4, optimizer=Adam, optimizer_params={'lr': 1e-4})
    agent = RainbowAgent(config)
