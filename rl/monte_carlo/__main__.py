from random import randrange
from rl.simulators import TicTacToe
from rl.monte_carlo import MonteCarlo


def main():

    mc = MonteCarlo(TicTacToe)

    state, action_mask = TicTacToe.reset()
    done = False
    steps = randrange(2)
    while True:
        TicTacToe.render(state)
        if steps % 2 == 0:
            state, action_mask, reward, done, _ = TicTacToe.step(state, mc.act(state, action_mask))
            if done:
                TicTacToe.render(state)
                print("Loss" if reward > 0 else "Draw")
                break
        else:
            action = int(input("Action: "))
            state, action_mask, reward, done, _ = TicTacToe.step(state, action)
            if done:
                TicTacToe.render(state)
                print("Win" if reward > 0 else "Draw")
                break
        steps += 1
    mc.close()


if __name__ == "__main__":
    main()
