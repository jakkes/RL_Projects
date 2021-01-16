import torch

@torch.jit.script
def choice(probabilities: torch.Tensor):
    cumsummed = (probabilities / probabilities.sum(-1, keepdim=True)).cumsum(-1)
    r = torch.rand(probabilities.shape[:-1]).unsqueeze_(-1)
    return (r > cumsummed).sum(-1)