from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

class Network():
    pass

class DDPGAgent():
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        # self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        pass
        # # Save experience in replay memory
        # self.memory.add(state, action, reward, next_state, done)
        
        # # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # if self.t_step == 0:
        #     # If enough samples are available in memory, get random subset and learn
        #     if len(self.memory) > BATCH_SIZE:
        #         experiences = self.memory.sample()
        #         self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        pass
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #     action_values = self.qnetwork_local(state)
        # self.qnetwork_local.train()

        # # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return np.argmax(action_values.cpu().data.numpy())
        # else:
        #     return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states 
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)

        # # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

    def soft_update():
        pass


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    print("Hello World")


def take_random_action():
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

if __name__ == '__main__':
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
    #take_random_action()
    main()