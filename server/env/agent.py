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
        transition = state + action + [reward] + next_state + [terminal_state]

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
    def __init__(self, n_states, n_actions):
        super(QFunctionNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states + n_actions, 5)
        self.linear2 = nn.Linear(5, 10)
        self.linear3 = nn.Linear(10, 10)
        self.linear4 = nn.Linear(10, 5)
        self.linear5 = nn.Linear(5, 1)

    def forward(self, input):
        output = F.relu(self.linear1(input))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))

        return self.linear5(output)

# QFunction -> in_features: state, action; out_features: E[sum(future outcomes)]

class Agent():
    def __init__(self):
        self.gamma = 0.9
        self.batch_size = 30
            
        self.get_possible_actions()

        self.QFunction = QFunctionNetwork(3*3, 14)
        self.QFunctionTarget = QFunctionNetwork(3*3, 14)
        self.QFunctionTarget.load_state_dict(self.QFunction.state_dict())
        self.QFunctionTarget.eval()
        
        self.optimizer = torch.optim.Adam(self.QFunction.parameters(), lr=0.001)
        self.mse_criterion = nn.MSELoss()
    
    # def get_expected_Q(self, state, action, reward, next_state, terminal):
    #     with torch.no_grad():
    #         state_action = torch.cat((state , action), 1)
    #             Q = self.QFunction(state_action.view(state_action.shape[0], 1, state_action.shape[1]))
    #         Q = self.QFunction()


    def get_possible_actions(self):
        valid = []
        
        valid.append([0.2, 0.15, 0.1, 0.25])
        valid.append([2, 3])
        valid.append([[False, False, False], [True, False, False], [False, True, False], [True, True, False],[False, False, True], [True, False, True], [False, True, True], [True, True, True]])
        valid.append([[1,1,1], [1,1,0], [1,0,0]])
        valid.append([[1,1,1], [1,1,0], [1,0,0]])
        valid.append([[1,1,1], [1,1,0], [1,0,0]])
        
        n_degrees = [len(v)-1 for v in valid]
        
        self.combinations = []
        counter = [0,0,0,0,0,0]
        i = 0

        while counter != n_degrees:
            for j in range(len(counter)-1):
                if counter[j] == n_degrees[j] + 1:
                    counter[j] = 0
                    counter[j+1] += 1
                elif counter[j] == n_degrees[j] + 2:
                    counter[j] = 1
                    counter[j+1] += 1
            i+=1
            self.combinations.append(counter.copy())
            counter[0] += 1
        
        self.actions = []

        for c in self.combinations:
            self.actions.append([valid[0][c[0]]] + [valid[1][c[1]]]  + valid[2][c[2]]  + valid[3][c[3]]  + valid[4][c[4]]  + valid[5][c[5]])

        self.actions = torch.tensor(self.actions)

    def random_action(self, n_actions):
        rows = self.actions.shape[0]
        ids = np.random.randint(rows, size=n_actions)
        return self.actions[ids, :].squeeze()

    def select_action(self, state, step):      
        coin = np.random.random()

        self.exploration = 0.9*np.exp(-1.0*step/50) + 0.05 # params from pytorch tutorial

        if coin < self.exploration:
            # random action
            action = self.random_action(1)
        else:
            with torch.no_grad():
                state = torch.tensor(state)*torch.ones((self.actions.shape[0], 1))
                state_action = torch.cat((state , self.actions), 1)
                
                Q = self.QFunction(state_action.view(state_action.shape[0], 1, state_action.shape[1]))
                id_action = torch.argmax(Q)
                
                action = state_action.squeeze()[id_action,9:] 
        return action.numpy()

    def fit(self, queue, batch_size, gamma):
        batch = queue.sample(batch_size)

        name2col = {"s": list(range(0,9)), "a": list(range(9, 23)), "r": 23, "s'": list(range(24, 33)), "d": 33}
                
        state = batch[:, :, name2col["s"]]    
        next_states = batch[:,:,name2col["s'"]]
        actions = batch[:,:,name2col["a"]]
        
        state_action = torch.cat((next_states, actions), 2)
        
        Q = self.QFunction(state_action).squeeze()
        ExpectedQ = batch[:, :, name2col['r']].squeeze() + gamma*(1-batch[:, :, name2col['d']].squeeze())*self.QFunctionTarget(state_action).squeeze()

        self.optimizer.zero_grad()
        loss = self.mse_criterion(Q, ExpectedQ)
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.QFunctionTarget.load_state_dict(self.QFunction.state_dict())

    def run(self, state):
        with torch.no_grad():
                state = torch.tensor(state)*torch.ones((self.actions.shape[0], 1))
                state_action = torch.cat((state , self.actions), 1)
                
                Q = self.QFunctionTarget(state_action.view(state_action.shape[0], 1, state_action.shape[1]))
                id_action = torch.argmax(Q)
                
                action = state_action.squeeze()[id_action,9:] 
        return action.numpy()
    def save(self, filename='model'):
        torch.save(self.QFunction.state_dict(), filename)
        torch.save(self.QFunctionTarget.state_dict(), filename+'target')

    
    def load(self, filename='model'):
        self.QFunction.load_state_dict(torch.load(filename))
        self.QFunctionTarget.load_state_dict(torch.load(filename+'target'))




# if __name__ == '__main__':
#     a = Agent()
#     q = ReplayBuffer(100)

#     #print(a.select_action([100, 100,100, 100, 11, 0, 2,3,4]))

#     for i in range(50):
#         population = [100, 90, 90]
        
#         names = ['CAT', 'MAD', 'AND']
#         sizes = [90, 100, 90]
        
#         UCI = [9, 10, 9]

#         print("Simulating")
#         sim = Simulator(population, names, sizes, UCI, 0, 2, 0.2)
#         sim.simulate(None)
#         print("Sim Done")
