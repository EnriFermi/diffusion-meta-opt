import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.s = deque(maxlen=capacity)
        self.a = deque(maxlen=capacity)
        self.r = deque(maxlen=capacity)
        self.s2 = deque(maxlen=capacity)
        self.d = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s2.append(s2)
        self.d.append(d)

    def __len__(self):
        return len(self.s)

    def sample(self, batch_size, device):
        idx = random.sample(range(len(self.s)), batch_size)
        s = torch.stack([self.s[i] for i in idx]).to(device)
        a = torch.stack([self.a[i] for i in idx]).to(device)
        r = torch.tensor([self.r[i] for i in idx], dtype=torch.float32, device=device)
        s2 = torch.stack([self.s2[i] for i in idx]).to(device)
        d = torch.tensor([self.d[i] for i in idx], dtype=torch.float32, device=device)
        return s, a, r, s2, d
