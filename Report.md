# Project Background: Why Multi-agent RL Matters

For artificial intelligence (AI) to reach its full potential, AI systems need to interact safely and efficiently with humans, as well as other agents. There are already environments where this happens on a daily basis, such as the stock market. And there are future applications that will rely on productive agent-human interactions, such as self-driving cars and other autonomous vehicles.

One step along this path is to train AI agents to interact with other agents in both cooperative and competitive settings. Reinforcement learning (RL) is a subfield of AI that shows promise for this application. However, much of RL's success to date has been in single agent domains, where building models that predict the behavior of other actors is unnecessary. As a result, traditional RL approaches, such as Q-Learning, are not well-suited for the complexity that accompanies environments where multiple agents are continuously interacting and evolving their policies.

# Multi-Agent Deep Deterministic Policy Gradient

This project trains two agents whose action space is continuous using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) reinforcement learning method called. The MADDPG algorithm, proposed by <a href="https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf">*Lowe et al.*</a>, is a relatively new concept in deep reinforecement learning (DRL).  MADDPG Is capable of finding the global maximum for a given environment and performs better than a number of other DRL algorithms such as DQN (Deep Q-Network), TRPO (Trust Region Policy Optimization) and DDPG (Deep Deterministic Policy Gradient). MADDPG is a type of Actor-Critic method, though unlike the DDPG algorithm, which trains each the actor and critic independently from one another, MADDPG trains both the actor and critic using information (actions and states) of all agents (though the trained actor can only act using its own state).

# Model architecture

### Actor network
The actor network, which maps states to actions, is a multi-layer neural network with the following architecture:
 - Input layer
 - Hidden layer (256 nodes, ReLu activation)
 - Batch normalization
 - Hidden layer (128 nodes, ReLu activation)
 - Output layer (tanh activation)

```
"""Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
        
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
```

### Critic network
The critic network, which  maps state-action pairs to Q-values, follows a similar architecture:

```
"""Critic (Value) Model."""

    def __init__(self, full_state_size, actions_size, seed=0, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+actions_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
```

# Training algorithm

The actor and critic networks are soft-updated and the agents use the current policy to explore the environment.  All episodes are saved into a shared replay buffer and each agent uses a minibatch (randomly selecting from the replay buffer) to train using the MADDPG algorithm. 

# Hyperparameters
The model was tested using a variety of hyperparameters; the following provides a set that successfully solves the environment:

```
BUFFER_SIZE              = int(1e5) # replay buffer size
BATCH_SIZE               = 256      # minibatch size
UPDATE_FREQ              = 1

GAMMA                    = 0.99     # discount factor
TAU                      = 1e-3     # for soft update of target parameters
LR_ACTOR                 = 1e-4     # learning rate of the actor 
LR_CRITIC                = 3e-4     # learning rate of the critic
WEIGHT_DECAY             = 0        # L2 weight decay

NOISE_REDUCTION_RATE     = 0.99
EPISODES_BEFORE_TRAINING = 500
NOISE_START              = 1.0
NOISE_END                = 0.1
```

# Results
Using the model architectures and hyperparameters outlined above, the environment was solved in 1,305 episodes; results are shown below.

```
0 episode	avg score 0.00000	max score 0.00000
100 episode	avg score 0.01610	max score 0.00000
200 episode	avg score 0.01070	max score 0.00000
300 episode	avg score 0.01650	max score 0.00000
400 episode	avg score 0.01470	max score 0.00000
500 episode	avg score 0.02040	max score 0.00000
600 episode	avg score 0.00090	max score 0.00000
700 episode	avg score 0.00000	max score 0.00000
800 episode	avg score 0.01410	max score 0.00000
900 episode	avg score 0.02280	max score 0.00000
1000 episode	avg score 0.09800	max score 0.10000
1100 episode	avg score 0.09110	max score 0.10000
1200 episode	avg score 0.12930	max score 0.10000
1300 episode	avg score 0.45530	max score 2.60000
Environment solved after 1305 episodes with the average score 0.5235000078566372

1400 episode	avg score 1.28350	max score 1.00000
1499 episode	avg score 0.93770	max score 0.10000
```

<img src="https://github.com/AaronChockla/tennis/blob/master/results.png" alt="Results" title="Results" style="max-width:100%;">


# Future Improvements
 - Improve model stability - the best results are only "reproducible" if the model is run numerous times (the model may fail to converge if run just once&mdash;or even a few times).
 - Implement the Proximal Policy Optimization (PPO), which seems to have worked for others on this environment.
 - Further hyperparameter tuning should be done to stablize the policy and address the sudden drop off ocne the high score is reached (only the beginnings of this phenomenon were observed here, as training was stopped after 1,500 episodes, but it is an issue that many have observed for this project and likely would appear if training were allowed to continue here). This may be accomplished, for example, by changing noise decay, learning rate, and batch size, along with the underlying network architecture itself.
 - Adding prioritized experience replay, rather than selecting experience tuples randomly, can improve learning by increasing the probability that rare or important experience vectors are sampled.
