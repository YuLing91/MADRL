import torch as T
import torch.nn.functional as F
from agent import Agent
import numpy as np

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    #保存模型
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    #加载模型
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    #用于在给定状态s的情况下选择动作a
    def choose_action(self, raw_obs):
        # raw_obs [[8个数],[10个数],[10个数]]
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx]) # [5个数]
            actions.append(action)
        #actions: [[5个数],[5个数],[5个数]]
        return actions
    
    # 智能体学习，更新价值网络Critic和策略网络Actor的参数
    def learn(self, memory):
        # 判断经验池的大小是否达到更新的标准，若是则进入第②步，若没有则结束
        if not memory.ready():
            return
        #数据抽样，从经验冲池进行采样，取batch_size大小的样本
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer() 
        
        # 将所有变量转为张量
        device = self.agents[0].actor.device
        states = T.tensor(states, dtype=T.float).to(device) # torch.Size([1024, 28])
        actions = T.tensor(actions, dtype=T.float).to(device) # torch.Size([3, 1024, 5])
        rewards = T.tensor(rewards).to(device) # torch.Size([1024, 3])
        states_ = T.tensor(states_, dtype=T.float).to(device) # torch.Size([1024, 28])
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        
        #获取每个智能体的数据
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])
        
        #把将两个张量（tensor）沿着某一维度拼接在一起，dim=1，按列拼接
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)# size:[1024,15]
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten() # size:1024
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten() # size:1024
            
            #TD算法更新价值网络Critic
            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            
            #梯度值由Pytorch自动计算，故我们只需要计算q-error(critic_loss)即可。
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            
            #梯度上升更新策略网络
            # 注意critic将(states,mu)作为输入
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            
            #更新target
            agent.update_network_parameters()
