from rl.simulators import ConnectFour


if __name__ == "__main__":
    state, _ = ConnectFour.reset()
    terminal = False
    while not terminal:
        print()
        ConnectFour.render(state)
        print()
        action = int(input("Action: "))
        state, _, _, terminal, _ = ConnectFour.step(state, action)
    ConnectFour.render(state)