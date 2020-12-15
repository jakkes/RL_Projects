import numpy as np

from simulators import TicTacToe


def test_steps():

    actions = [4, 0, 2, 6, 3, 8, 5]

    state = TicTacToe.reset(1).ravel()
    terminal = False
    reward = 0.0
    player = 1.0
    for action in actions:
        assert reward == 0.0
        assert not terminal
        state, reward, terminal, _ = TicTacToe.step(state, action)
        assert state[action] == player
        player = -player

    assert reward == 1.0
    assert terminal


def test_diag_win():
    states = np.array([[1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0]])
    states, rewards, terminals, _ = TicTacToe.step_bulk(states, np.array([8, 6], dtype=np.int32))

    assert all(terminals)
    assert np.all(rewards == 1.0)
