import torch
from torch.multiprocessing import Process, Queue
from torch import Tensor

from rl.utils.env import repeat_action

class Worker(Process):
    def __init__(self, conn: Queue):
        super().__init__(daemon=True)
        self.conn: Queue = conn
        
    def run(self):
        env = self.conn.get()
        actor = self.conn.get()
        critic = self.conn.get()
        opt = self.conn.get()
        opt_params = self.conn.get()
        batchsize = self.conn.get()
        action_repeats = self.conn.get()
        discount = self.conn.get()

        opt = opt(
            list(actor.parameters()) + list(critic.parameters()),
            **opt_params
        )

        logprobs = torch.zeros(batchsize)
        deltas = torch.zeros(batchsize)
        i = 0

        while True:
            
            done = False
            state = torch.as_tensor(env.reset()).view(1, -1)

            while not done:
                action_probabilities: Tensor = actor(state).squeeze()
                action = torch.sum(action_probabilities.cumsum(0) < torch.rand((1, ))).item()

                next_state, reward, done, _ = repeat_action(env, action, action_repeats)
                next_state = torch.as_tensor(next_state).view(1, -1)

                with torch.no_grad():
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + discount * critic(next_state)[0, 0]

                logprobs[i] = action_probabilities[action].log()
                deltas[i] = td_target - critic(state)[0, 0]

                i += 1
                if i >= batchsize:
                    opt.zero_grad()
                    (deltas.pow(2).div_(2) - deltas.detach() * logprobs).mean().backward()
                    opt.step()
                    logprobs = torch.zeros_like(logprobs)
                    deltas = torch.zeros_like(deltas)
                    i = 0

                state = next_state