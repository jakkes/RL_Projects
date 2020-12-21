import torch

from .. import _next_board, _distance_to_empty, _get_suicidemask
from .. import Go


def test_next_board():
    board = torch.zeros(2, 3, 3)

    next_boards1 = _next_board(board, 0, torch.arange(9))
    next_boards2 = _next_board(board, 1, torch.arange(9))

    true_next_boards1 = torch.zeros(9, 2, 3, 3).view(9, 2, 9)
    true_next_boards1[torch.arange(9), 0, torch.arange(9)] = 1.0
    true_next_boards1 = true_next_boards1.view(9, 2, 3, 3)

    true_next_boards2 = torch.zeros(9, 2, 3, 3).view(9, 2, 9)
    true_next_boards2[torch.arange(9), 1, torch.arange(9)] = 1.0
    true_next_boards2 = true_next_boards2.view(9, 2, 3, 3)

    assert (next_boards1 == true_next_boards1).all()
    assert (next_boards2 == true_next_boards2).all()


def test_distance_to_empty():
    boards = torch.tensor([
        [[[0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0]],

        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]]],


        [[[1.0, 1.0, 0.0],
         [1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],

        [[0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]]]
    ])

    distances = _distance_to_empty(boards, 0)
    true_distances = torch.tensor([
        [[-1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]],

        [[3.0, 2.0, -1.0],
        [2.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0]]
    ])

    assert (true_distances == distances).all()


def test_suicide_mask():
    board1 = torch.tensor([
        [[0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0]],

        [[0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]]
    ])

    mask1 = _get_suicidemask(board1, 0)
    true1 = torch.tensor(
        [[True, False, False, False, False],
        [False, False, False, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]]
    )

    assert (mask1 == true1).all()


def test_go():
    env = Go(5)
    state = env.reset()

    assert (state == torch.zeros(2, 5,5)).all()

    action_sequence = [6, 1, 7, 5, 2, 0, 10]

    state_sequence = [
        torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ]),

        torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        ])
    ]
    
    for action, true_state in zip(action_sequence, state_sequence):
        state, reward, done = env.step(action)

        assert (state == true_state).all()
    
    env.step(25)
    _, reward, done = env.step(25)

    assert done
    assert reward == 1

def test_repeated_state():
    env = Go(5)
    env.reset()
    env.step(1)
    env.step(2)
    env.step(5)
    env.step(6)
    env.step(11)
    env.step(12)
    env.step(15)
    env.step(8)
    env.step(7)

    val = False
    try:
        env.step(8)
    except AssertionError:
        val = True

    assert val

def run_tests():
    test_next_board()
    test_distance_to_empty()
    test_suicide_mask()
    test_go()
    test_repeated_state()