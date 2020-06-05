# Deep Reinforcement Learning: Continuous Control (Report)

[![Twitter Follow](https://img.shields.io/twitter/follow/youldash.svg?style=social?style=plastic)](https://twitter.com/youldash)

## License

By using this site, you agree to the **Terms of Use** that are defined in [LICENSE](https://github.com/youldash/DRL-Continuous-Control/blob/master/LICENSE).

## Algorithm Implementation

As mentioned in the [`README.md`](https://github.com/youldash/DRL-Continuous-Control/blob/master/README.md) file of this repo, The project was developed in partial fulfillment of the requirements for Udacity's [Deep Reinforcement Learning (DRL) Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. To solve the challenges presented therein, we explored (and implemented) the [Deep Deterministic Policy Gradient (DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) algorithm. This choice is motivated by the fact that the action space is continuous, and the DDPG algorithm has shown quite an impressive performance in past.

## The Deep Deterministic Policy Gradient Algorithm

Initial attempts were made for developing `agent` implementations of the **DDPG** algorithm (see `agent.py` for details). The algorithm is summarized below:

[ddpg]: misc/DDPG.png "Deep Deterministic Policy Gradient (DDPG)."

<div align="center">
	<img src="misc/DDPG.png" width="75%" />
</div>






### Early Attempts

- Our first `model` training configuration was based on a DQN with **two Fully-connected (FC) layers (hosting 512 nodes in each layer)**. This `model` configuration solved the virtual world (or environment) in a number of episodes that far exceeded 700. This was set as a point of reference to beat in our next attempts. The `model`'s [Neural Network (NN)](https://pathmind.com/wiki/neural-network) architecture was adjusted based on the following configuration:
```
Input nodes (37) -> FC Layer (512 nodes, ReLU activation) -> FC Layer (512 nodes, ReLU activation) -> Output nodes (4)
```

- Further attempts were made by amending the `model` architecture (*i.e.* by increasing the number of layers, as well as increasing the number of nodes in the `model`). These experiments yielded **poorer** results when compared to the **benchmark** configuration above, leading us to further adjust the `model` by having **two FC layers (having 128 nodes in the first, and 32 nodes in the second)**. This architecture solved the environment in less than 500 episodes. This architecture is summarized as the following:
```
Input nodes (37) -> FC Layer (128 nodes, ReLU activation) -> FC Layer (32 nodes, ReLU activation) -> Output nodes (4)
```
> At this point it is imperative to note that this particular `model` configuration was considered as the foundation to be used in our future tests of the agent (for further comparisons), since the results looked promising. See the [`NavigationUsingDQN.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDQN.ipynb) notebook for implementation details and the rewards (*i.e.* the results) obtained after training and testing.

### Rewards Plot

The following graph (*i.e.* plot) illustrated two measurable outcomes. These are

- the rewards per-episode within the training phase, in addition to
- the moving mean.

![](./plots/RewardsUsingDQN.png)

The trained agent, as witnesses in the accompanying [`NavigationUsingDQN.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDQN.ipynb) notebook file, revealed the following results:

```
EPISODE 100	AVG SCORE: 9.6000	EPS: 0.0270	LEARNING RATE: [0.00022711322607504007]
EPISODE 174	AVG SCORE: 13.0200	EPS: 0.0104	LEARNING RATE: [0.00013037578502107008]

Environment solved in 74 episodes.
Average score: 13.02.
Model saved successfully.

Solved in 3.63 minutes.
```

## The Double Deep Q-Network Algorithm

After we concluded our tests using the baseline **DQN** algorithm, we sought to include a version of the **Double Deep Q-Network (Double-DQN)** algorithm. According to [literature](https://arxiv.org/abs/1509.06461), this approach aims to reduce the overestimations that are otherwise detected in the aforementioned approach (*i.e.* the vanilla **DQN**) while training in game simulations/environments. See `model.py` for more details.

### Attempts Using Double-DQNs

- Using the **benchmark** `model` configuration, which was revealed above, we strived to achieve better results that those obtained by the previous attempts. However, that was not the case.

- See the [`NavigationUsingDoubleDQN.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDoubleDQN.ipynb) notebook for implementation details and the rewards (*i.e.* the results) obtained after training and testing.

### Rewards Plot

The following graph illustrated the outcomes:

![](./plots/RewardsUsingDoubleDQN.png)

The trained agent, as witnesses in the accompanying   [`NavigationUsingDoubleDQN.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDoubleDQN.ipynb) notebook file, revealed the following results:

```
EPISODE 100	AVG SCORE: 10.6100	EPS: 0.0270	LEARNING RATE: [0.00022711322607504007]
EPISODE 182	AVG SCORE: 13.0300	EPS: 0.0100	LEARNING RATE: [0.00012278292199717212]

Environment solved in 82 episodes.
Average score: 13.03.
Model saved successfully.

Solved in 3.71 minutes.
```

Here, the number of episodes exceeded the **benchmark** by 8 episodes only.

## The Dueling Q-Network Algorithm

After we concluded our previous tests using the **Double-DQN** algorithm, we sought to accommodate a version of the **Dueling Q-Network** which is commonly abbreviated as a **Dueling Network** (or **DN** for short). Yet after all attempts (and surprisingly enough), we never reached a stage of confidence that would settle this approach as being better than the previous attempts.

### Attempts Using DNs

- Using the **benchmark** `model` configuration, which was revealed above, we strived to achieve better results that those obtained by the previous attempts. However, that was not the case.

- See the [`NavigationUsingDuelingNetwork.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDuelingNetwork.ipynb) notebook for implementation details and the rewards (*i.e.* the results) obtained after training and testing.

### Rewards Plot

The following graph illustrated the outcomes:

![](./plots/RewardsUsingDuelingNetwork.png)

The trained agent, as witnesses in the accompanying   [`NavigationUsingDuelingNetwork.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDuelingNetwork.ipynb) notebook file, revealed the following results:

```
EPISODE 100	AVG SCORE: 8.7700	EPS: 0.0270	LEARNING RATE: [0.00022711322607504007]
EPISODE 200	AVG SCORE: 11.8600	EPS: 0.0100	LEARNING RATE: [0.00010727666846424718]
EPISODE 244	AVG SCORE: 13.1000	EPS: 0.0100	LEARNING RATE: [7.712247038635614e-05]]

Environment solved in 144 episodes.
Average score: 13.1.
Model saved successfully.

Solved in 6.34 minutes.
```

Here, the number of episodes are significantly higher, and the time taken to solve the environment took much longer than anticipated.

## Combining Double-DQNs With Dueling Networks

After concluding our tests using the **DN** architecture, we sought to accommodate a version that would combine both **Double-DQNs** and **DNs** in one reliable architecture to see whether this combination might actually end up outperforming all the previous test runs. To achieve this, we enabled/allowed ALL of our notebooks to globally set the appropriate parameters/flags prior training and testing.

### Attempts Using Double-DQNs + DNs

- Using the **benchmark** `model` configuration, we strived to achieve better results that those obtained by the previous attempts. As a result, we achieved outcomes that are considered better than the attempts made by the previous **DN** implementation (yet not better than the plain-vanilla **DQN** approach in terms of the number of episodes noted).

- In the [`NavigationUsingDoubleDQNWithDuelingNetwork.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDoubleDQNWithDuelingNetwork.ipynb) notebook file you can see that there are three flags that can be toggled (on or off). The following snippet depicts these three variables:

``` Python
""" Global configuration.
"""
TOGGLE_DOUBLE_DQN = True            # True for the Double-DQN method. False for the fixed Q-target method.
TOGGLE_DUELING_NETWORK = True       # True for the Dueling Network (DN) method.
TOGGLE_PRIORITIZED_REPLAY = False   # True for the Prioritized Replay memory buffer.
```

### Rewards Plot

The following graph illustrated the outcomes:

![](./plots/RewardsUsingDoubleDQNWithDuelingNetwork.png)

The trained agent, as witnesses in the accompanying   [`NavigationUsingDoubleDQNWithDuelingNetwork.ipynb`](https://github.com/youldash/DRL-Continuous-Control/blob/master/NavigationUsingDoubleDQNWithDuelingNetwork.ipynb) notebook file, revealed the following results:

```
EPISODE 100	AVG SCORE: 10.3200	EPS: 0.0270	LEARNING RATE: [0.00022711322607504007]
EPISODE 158	AVG SCORE: 13.0600	EPS: 0.0128	LEARNING RATE: [0.00014699916918461692]

Environment solved in 58 episodes.
Average score: 13.06.
Model saved successfully.

Solved in 4.06 minutes.
```

Although the environment was solved in lesser time as we compared it against the plain **DN** approach, the number of episodes reached were a little higher than the **benchmark** `model` configuration.

## Parameter Tuning

In all of our experiments a set of tuning parameters (or rather **hyperparameters**) enabled us to explore the possible variations possible for achieving the results (both reported here, and others expected in future tuning attempts). Ideally, it is worthy to mention that one single hyperparameter configuration might work with one `model`, and may well **NOT** be suitable with others.

### Notebook Parameters

In all the Jupyter notebooks of tis repo you may tweak the following parameters:

``` Python
def dqn(n_episodes=2e3, max_t=int(1e3), eps_start=1., eps_end=1e-2, eps_decay=995e-3):
    """ Implementation of the Deep Q-Network (DQN) algorithm.
    
    Params
    ======
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon (ε), for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon (ε)
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon
    """
```

In addition to the above parameters the following parameter segments are also adjustable (see the `agent.py` Python script for details):

``` Python
""" Hyperparameter setup.
"""
BUFFER_SIZE = int(1e5)  # Replay buffer size.
BATCH_SIZE = 64         # Minibatch size.
LEARNING_RATE = 4.8e-4  # Learning rate.
THRESHOLD = 4           # How often to update the network.
GAMMA = 99e-2           # Discount factor.
TAU = 1e-2              # For soft update of target parameters.
```

``` Python
double_dqn = False
""" True for the Double-DQN method.
"""

dueling_network = False
""" True for the Dueling Network (DN) method.
"""

prioritized_replay = False
""" True for the Prioritized Replay memory buffer.
"""
```

``` Python
def __init__(
    self, state_size, action_size, seed, lr_decay=9999e-4,
    double_dqn=False, dueling_network=False, prioritized_replay=False):
    """ Initialize an Agent instance.
    
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        lr_decay (float): Multiplicative factor of learning rate decay
        double_dqn (bool): Toogle for using the Double-DQN method
        dueling_network (bool): Toogle for using the Dueling Network (DN) method
        prioritized_replay (bool): Toogle for using the Prioritized Replay method
    """
```

## Conclusion and Future Work

This report presented out work in training an agent to solve the environment, while considering varying architectures to determine which `model` configuration would be deemed the best in our experiments. In all attempts the results shown in the [first part](https://github.com/youldash/DRL-Continuous-Control/blob/master/REPORT.md#the-deep-q-network-algorithm) set the benchmark for future attempts. 

The work presented herein was made possible using Udacity's [NVIDIA Tesla K80 accelerator GPU](https://www.nvidia.com/en-gb/data-center/tesla-k80/) architecture. In future works we plan on implementing (and training all agents) locally using NVIDIA GPUs (both internally, and externally mounted using eGPU enclosures).

Although the [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) memory buffer algorithm was mentioned above, we did not fully test its implementation and made this a possible "future work" attempt.
 
With the possibility of reaching better outcomes by tweaking the parameters a bit (as seen above), perhaps it it noteworthy to include other state-of-the-art algorithm implementations and compare them against our **benchmark**. These of which include (and are not limited to) the following:

* [Distributional-DQNs](https://arxiv.org/abs/1707.06887) by Marc G. Bellemare, Will Dabney, and Rémi Munos.
* [Noisy Networks](https://arxiv.org/abs/1706.10295) by Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and Shane Legg.
* [Asynchronous Methods for DRL](https://arxiv.org/abs/1602.01783) by Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu.
* And others...
