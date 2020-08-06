from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor, LongTensor, Size

@torch.jit.script
def _update_normalization(values, minmaxdata):
    mi = values.min()
    ma = values.max()

    if mi < minmaxdata[0]:
        minmaxdata[0] = mi
    
    if ma > minmaxdata[1]:
        minmaxdata[1] = ma

@torch.jit.script
def _normalize_Q(Q, minmaxdata):
    if minmaxdata[1] < minmaxdata[0]:
        return Q
    return ((Q - minmaxdata[0]) / (minmaxdata[1] - minmaxdata[0])).clamp_(0, 1)

@torch.jit.script
def _select(nodes, P, Q, N, mask, children, minmax, c1: float, c2: float):
    nodes = nodes[mask]
    children = children[mask, nodes]
    bv = torch.arange(mask.sum())
    bvr = bv.repeat_interleave(children.shape[-1])
    P = P[mask]; Q = Q[mask]; N = N[mask]
    P = P[bvr, children.view(-1)].view(children.shape)
    Q = _normalize_Q(Q[bvr, children.view(-1)].view(children.shape), minmax)
    N = N[bvr, children.view(-1)].view(children.shape)

    coeff = (((N.sum(1, keepdim=True) + c2 + 1) / c2).log_() + c1) * N.sum(1, keepdim=True).sqrt_()
    actions = (Q + P / (1 + N) * coeff).argmax(1)
    return actions, children[bv, actions]

@torch.jit.script
def _backup(nodes, values, Q, N, R, parents, discount, bv, bvr, minmax):
    G = values.clone()

    while True:
        mask = parents[bv, nodes] > -1
        if mask.sum() == 0:
            break
        G[mask] = R[mask, nodes[mask]] + discount * G[mask]
        N[mask, nodes[mask]] += 1
        Q[mask, nodes[mask]] += 1.0 / N[mask, nodes[mask]] * (G[mask] - Q[mask, nodes[mask]])
        nodes[mask] = parents[mask, nodes[mask]]
        _update_normalization(G[mask], minmax)

def _get_children_indices(simulation: int, action_dim: int) -> LongTensor:
    return (1 + simulation * action_dim) + torch.arange(action_dim)

def run_mcts(root_states: Tensor, simulations: int, agent: '.agent.MuZeroAgent'):

    minmax = torch.tensor([np.inf, -np.inf])

    priors, values = agent.prediction_net(root_states)

    bn = root_states.shape[0]        # batch number
    action_dim = priors.shape[-1]
    
    bv = torch.arange(bn)       # batch vec 
    bvr = bv.repeat_interleave(action_dim)      # batch vec repeated

    P = torch.zeros(bn, 1 + (simulations + 1) * action_dim)
    Q = torch.zeros(bn, 1 + simulations * action_dim)
    N = Q.clone(); R = N.clone()
    states = torch.zeros(Q.shape + root_states.shape[1:])
    children = torch.empty(bn, 1 + simulations * action_dim, action_dim, dtype=torch.long).fill_(-1)
    parents = torch.empty(bn, 1 + (simulations + 1) * action_dim, dtype=torch.long).fill_(-1)
    
    states[:, 0] = root_states
    P[:, 1:action_dim+1] = priors
    ci = _get_children_indices(0, action_dim)
    children[:, 0] = ci.view(1, -1).repeat(bn, 1)
    parents[:, ci] = 0

    for s in range(simulations):
        nodes = torch.zeros(bn, dtype=torch.long)
        actions = torch.zeros_like(nodes)
        while True:
            mask = children[bv, nodes, 0] > -1
            if mask.sum() == 0:
                break

            selected_actions, selected_children = _select(nodes, P, Q, N, mask, children, minmax, agent.config.c1, agent.config.c2)
            nodes[mask] = selected_children
            actions[mask] = selected_actions

        ci = _get_children_indices(s+1, action_dim)     # child indices
        cir = ci.repeat(bn)
        children[bv, nodes] = cir.view(nodes.shape[0], -1) #ci.view(1, -1).repeat(bn, 1)
        parents[bvr, cir] = nodes.repeat_interleave(action_dim)

        new_states, rewards = agent.dynamics_net(states[bv, parents[bv, nodes]], actions)
        states[bv, nodes] = new_states
        R[bv, nodes] = rewards.squeeze()

        priors, values = agent.prediction_net(states[bv, nodes])
        P[bvr, cir] = priors.view(-1)
        values = values.view(-1)

        _backup(nodes, values, Q, N, R, parents, agent.config.discount, bv, bvr, minmax)

    return P[:, 1:1+action_dim], Q[:, 1:1+action_dim], N[:, 1:1+action_dim]
