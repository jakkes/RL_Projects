import torch

@torch.jit.script
def f(x):
    for i, j in zip([1, 4, -1, 4], [4,2,1,2]):
        if x > 0 or x < 1:
            x += i
            x += j
    return x
    
if __name__ == "__main__":
    print(f(torch.tensor(5.0)))