import torch
from torch.multiprocessing import Process, Queue

from .utils import repeat_action

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

        steps = 0
        loss = torch.tensor(0.0)
        while True:
            
            done = False
            state = torch.as_tensor(env.reset()).view(1, -1)

            while not done:
                steps += 1
                action_probabilities: Tensor = actor(state).squeeze()
                action = torch.sum(action_probabilities.cumsum(0) < torch.rand((1, ))).item()

                next_state, reward, done, _ = repeat_action(env, action, action_repeats)
                next_state = torch.as_tensor(next_state).view(1, -1)

                with torch.no_grad():
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + discount * critic(next_state)[0, 0]

                delta = td_target - critic(state)[0, 0]
                loss += delta.pow(2) - delta.detach() * action_probabilities[action].log()

                if steps >= batchsize:
                    loss /= batchsize
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss = 0
                    steps = 0

                state = next_state