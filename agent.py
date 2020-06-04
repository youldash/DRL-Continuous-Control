# Imports.
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim # For optimizer support.

# Other imports.
from model import Actor, Critic
from buffer import ReplayBuffer
from noise import Noise
from collections import namedtuple, deque


""" Hyperparameter setup.
"""
BUFFER_SIZE = int(1e6)  # Replay buffer size (5e5 | 1e6).
BATCH_SIZE = 1024       # Minibatch size (128 | 256 | 512 | 1024).
LR_ACTOR = 1e-4         # Learning rate of the Actor (1e-3 | 1e-4).
LR_CRITIC = 1e-3        # Learning rate of the Critic (1e-3 | 1e-4).
GAMMA = 99e-2           # Discount factor.
TAU = 1e-3              # For soft update of target parameters.
WEIGHT_DECAY = 0.       # L2 weight decay.
# NUM_TIME_STEPS = 20     # How often to update the network (i.e. time steps).
# NUM_LEARN_UPDATES = 10  # Number of learning updates (used in step()).


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
""" Set the working device on the NVIDIA Tesla K80 accelerator GPU (if available).
    Otherwise we use the CPU (depending on availability).
"""


class Agent():
    """ Class implementation of a so-called "intelligent" agent.
        This agent interacts with and learns from the environment.
        This agent employs the DDPG algorithm to solve this problem.
    """

    # actor_local = None
    # actor_target = None
    # actor_optimizer = None
    """ Class-level Actor properties.
    """

    # critic_local = None
    # critic_target = None
    # critic_optimizer = None
    """ Class-level Critic properties.
    """

    # memory = None
    """ Class-level memory variable.
    """
        
    
    def __init__(self, state_size, action_size, seed, add_noise=True):
        """ Initialize an Agent instance.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            add_noise (bool): Toggle for using the stochastic process
        """

        # Set the parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Setting the Actor network (with the Target Network).
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        # Optimize the Actor using Adam.
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
                
        # Setting the Critic network (with the Target Network).
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        
        # Optimize the Critic using Adam.
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Set up noise processing.
        if add_noise:
            self.noise = Noise((20, action_size), seed)
        
        # Use the Replay memory buffer (once per class).
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
                
        # Initialize the time step (until max NUM_TIME_STEPS is reached).
        # self.t_step = 0


    def step(self, time_step, states, actions, rewards, next_states, dones):
        """ Update the network on each step.
            In other words, save the experience in replay memory,
            and then use random sampling from the buffer to learn.
        """

        # Save experience in replay memory.
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done) 
        
        # Learn every time step till NUM_TIME_STEPS is reached.
        # if time_step % NUM_TIME_STEPS != 0:
        #     return
        
        # Save the experience in replay memory, then use random sampling from the buffer to learn.
        self.sample_and_learn()
        
        
    def sample_and_learn(self):
        """ For a specified number of agents,
            use random sampling from the buffer to learn.
        """
        
        # If enough samples are available in memory, get random subset and learn.
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
            # for _ in range(NUM_LEARN_UPDATES):
            #     experiences = Agent.memory.sample()
            #     self.learn(experiences, GAMMA)

    
    def act(self, state, add_noise=True):
        """ Return the actions for a given state as per current policy.
        
        Params
        ======
            state (array_like): Current state
            add_noise (bool): Toggle for using the stochastic process
        """
            
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # If the stochastic process is enabled.
        if add_noise:
            action += self.noise.sample()
                
        # Return the action.
        return np.clip(action, -1, 1)
                

    def reset(self):
        """ Reset the state.
        """
        
        # Reset the internal state (noise) to mean (mu).
        self.noise.reset()
        
        
    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.
            i.e.,
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where
                actor_target(state) -> action, and
                critic_target(state, action) -> Q-value.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done, w) tuples 
            gamma (float): Discount factor
        """

        # Set the parameters.
        states, actions, rewards, next_states, dones = experiences
        
        """ Update the Critic.
        """
        # Get the predicted next-state actions and Q-values from the target models.
        # Calculate the pair action/reward for each of the next_states.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q-targets for the current states, (y_i).
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute the Critic loss.
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss.
        self.critic_optimizer.zero_grad()        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        """ Update the Actor.
        """
        # Compute the Actor loss.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss.
        self.actor_optimizer.zero_grad()        
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        actor_loss.backward()
        self.actor_optimizer.step()

        """ Update the target networks.
        """
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
            i.e.,
            θ_target = τ * θ_local + (1 - τ) * θ_target.

        Params
        ======
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)
