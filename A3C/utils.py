from gym import Env

def repeat_action(env: Env, action: int, repeats: int):
    reward = 0.0
    for _ in range(repeats):
        next_state, r, done, info = env.step(action)
        reward += r
        if done:
            break
    return next_state, reward, done, info