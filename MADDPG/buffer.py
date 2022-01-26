import numpy as np

# MultiAgentReplayBuffer就是用来存储一系列等待学习的SARS片段。
class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        self.mem_size = max_size #经验池保存的个数：1000000
        self.mem_cntr = 0 #经验池中的当前索引
        self.n_agents = n_agents #智能体个数
        self.actor_dims = actor_dims #[8,10,10]
        self.batch_size = batch_size  #更新批量的大小
        self.n_actions = n_actions #动作空间维度
        
        #创建初始化memory 对转移元组的每一元素都有一个重放容器，这样便于分开管理，而不用记录同一元组的下标顺序
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool) #为了方便训练，存储了“完成标记”（done），用于定义终态的价值为0

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    #用于将（s, a, r, s_）对存储于记忆池中 
    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):
        
        #只需将容器计数指针mem_cntr % mem_size处写入数据即可，完成后，将指针后移一位。
        #由于指针数值可能会超过容器大小，因此需要使用模%运算来覆盖掉之前写入的数据。
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    #在经验复用池中采样，，采样的大小是batch_size
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        #采样
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        #采样batch数据
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        #当重放缓冲区中的数据大于等于批处理时，才开始学习
        if self.mem_cntr >= self.batch_size:
            return True
