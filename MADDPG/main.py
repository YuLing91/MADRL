import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import make_env

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    
    #scenario = 'simple'
    scenario = 'simple_adversary' #环境名称
    env = make_env(scenario)
    n_agents = env.n # 智能体个数 n_agents 3
    
    #定义状态空间，动作空间
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    
    critic_dims = sum(actor_dims) #actor_dims:[8, 10, 10],critic_dims:28
    
    # 动作空间是一个数组列表，假设每个智能体都有相同的动作空间
    n_actions = env.action_space[0].n # n_actions:5
    
    ##初始化Agent 用MADDPG算法
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')
    
    #memory用于储存跑的数据的数组，经验池保存的个数：1000000
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000  #训练的总次数
    MAX_STEPS = 25  #设置一个回合最大步数
    total_steps = 0
    score_history = [] #用于记录每个EP的reward，统计变化
    evaluate = False  #是否评估模型
    best_score = 0
    
    # 如果是评估模型，直接加载模型的参数
    if evaluate:
        maddpg_agents.load_checkpoint()
    
    # 训练模型 主流程
    for i in range(N_GAMES):
         # ① 重置状态
        obs = env.reset() # 初始化环境 [8,10,10]
        score = 0 #记录当前EP的reward
        done = [False]*n_agents # 定义3个False数组[False, False, False]
        episode_step = 0
        while not any(done):
            if evaluate: 
                env.render() # 刷新当前环境，并显示
                #time.sleep(0.1) # 放慢视频的动作
            # ② 选择动作 把obs整理一下，放入Actor网络，输出actions。
            actions = maddpg_agents.choose_action(obs) # [[5个数],[5个数],[5个数]]
            # ③ 与环境互动，获得obs_, reward, done, info数据
            obs_, reward, done, info = env.step(actions)# obs_:[[8个数],[10个数],[10个数]],reward:[3个数]
            state = obs_list_to_state_vector(obs) # [28个数]
            state_ = obs_list_to_state_vector(obs_) # [28个数]
            
            # 如果达到最大迭代步数，就结束该回合的训练
            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            # ④ 保存数据
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            # ⑤ 如果数据量足够，就对数据进行随机抽样，更新Actor和Critic
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)
                
            # ⑥ 把obs_赋值给obs，开始新的一步
            obs = obs_

            score += sum(reward) #记录当前EP的总reward
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
