# Snake
Not part of the Gym library but behaves similarly to environments there.

## Usage instructions
```python
from snake import Snake
env = Snake(version=0)                          # Versions allowed are 0 and 1
initial_state = env.reset()                     # Must be called initially and if done=True is returned
next_state, reward, done, _ = env.step(2)       # Allowed actions are 0, 1, 2, 3
```
The states are boolean tensors of shape `(2, W, H)` where `W = H = 5` for version 0 and `W = H = 20` for version 1.

## Reward structure
Rewards are given in the following way (exclusively, i.e. only one of the following is applied):  
- If agent eats: +10
- If agent dies: -10
- If agent moves towards apple: +0.1
- If agent moves away from apple: -0.1