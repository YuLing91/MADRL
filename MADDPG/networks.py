import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Critic批评者，在DDPG中，接受来自Actor的一个Action值和当前的状态，输出的是当前状态下，采用Action动作以后得到的关于Q的期望。
class CriticNetwork(nn.Module):
    #input_dims:状态+动作数，fc1_dims：第一层的神经元个数，fc2_dims：第二层的神经元个数，n_actions：动作数
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        # 按维数1拼接 T.cat([state, action], dim=1)即行数不变，列数增加
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


# Actor作用是接收状态描述，输出一个action，由于DDPG中的动作空间要求是连续的，所以使用了一个softmax
class ActorNetwork(nn.Module):
    #input_dims:状态数，fc1_dims：第一层的神经元个数，fc2_dims：第二层的神经元个数，n_actions：动作数
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

