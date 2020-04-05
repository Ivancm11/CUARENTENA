import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminal_state):
        """"
            Save transition
            Input must be arrays
        """
        transition = state + [action, reward] + next_state + [terminal_state]

        if len(self.buffer) < self.size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        episodes = [self.buffer[i] for i in indexes]
        episodes = torch.tensor(episodes)
        return episodes.view(batch_size, 1, -1)

    def __len__(self):
        return len(self.buffer)

class QFunctionNetwork(nn.Module):
    def __init__(self):
        super(QFunctionNetwork, self).__init__()
        self.linear1 = nn.Linear(3, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, input):
        output = F.relu(self.linear1(input))
        return self.linear2(output)

# QFunction -> in_features: state, action; out_features: E[sum(future outcomes)]
QFunction = QFunctionNetwork()
QFunctionTarget = QFunctionNetwork()