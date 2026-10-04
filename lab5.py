# ============================================================
# PROGRAM 1: Install and configure Gymnasium
# ============================================================

# Run these commands in Jupyter/Terminal if libraries are not installed:
# !pip install gymnasium
# !pip install torch
# !pip install matplotlib
# !pip install numpy
# !pip install pandas

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

print("Libraries configured successfully.")


# ============================================================
# PROGRAM 2: Create and execute a simple RL environment
# ============================================================

import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

state, info = env.reset()

print("Initial State:", state)

action = env.action_space.sample()

next_state, reward, terminated, truncated, info = env.step(action)

print("Action:", action)
print("Next State:", next_state)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()


# ============================================================
# PROGRAM 3: Explore observation space and action space
# ============================================================

import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

print("Observation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

print("\nNumber of States:")
print(env.observation_space.n)

print("\nNumber of Actions:")
print(env.action_space.n)

env.close()


# ============================================================
# PROGRAM 4: Display states, actions, rewards and termination
# ============================================================

import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

state, info = env.reset()

print("Initial State:", state)

for step in range(10):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step + 1)
    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("Next State:", next_state)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    state = next_state

    if terminated or truncated:
        break

env.close()


# ============================================================
# PROGRAM 5: Simulate random actions in FrozenLake
# ============================================================

import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

number_of_episodes = 10

for episode in range(number_of_episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(100):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        state = next_state

        if terminated or truncated:
            break

    print(
        "Episode:",
        episode + 1,
        "Total Reward:",
        total_reward
    )

env.close()


# ============================================================
# PROGRAM 6: Implement Q-Learning for FrozenLake
# ============================================================

import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

q_table = np.zeros(
    (number_of_states, number_of_actions)
)

learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000

for episode in range(episodes):

    state, info = env.reset()

    for step in range(100):

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        best_next_action = np.max(
            q_table[next_state]
        )

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward
            + gamma * best_next_action
            - q_table[state, action]
        )

        state = next_state

        if terminated or truncated:
            break

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

env.close()

print(q_table)


# ============================================================
# PROGRAM 7: Initialize and update Q-Table
# ============================================================

import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

states = env.observation_space.n
actions = env.action_space.n

q_table = np.zeros(
    (states, actions)
)

print("Initial Q-Table:")
print(q_table)

state, info = env.reset()

action = env.action_space.sample()

next_state, reward, terminated, truncated, info = env.step(
    action
)

learning_rate = 0.8
gamma = 0.95

old_value = q_table[state, action]

next_max = np.max(
    q_table[next_state]
)

new_value = old_value + learning_rate * (
    reward
    + gamma * next_max
    - old_value
)

q_table[state, action] = new_value

print("\nUpdated Q-Table:")
print(q_table)

env.close()


# ============================================================
# PROGRAM 8: Train agent for multiple episodes
# ============================================================

import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

states = env.observation_space.n
actions = env.action_space.n

q_table = np.zeros(
    (states, actions)
)

learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000

episode_rewards = []

for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(100):

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        q_table[state, action] += learning_rate * (
            reward
            + gamma * np.max(q_table[next_state])
            - q_table[state, action]
        )

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

env.close()

print("Training completed.")


# ============================================================
# PROGRAM 9: Display learned Q-Table
# ============================================================

print("Learned Q-Table:")
print(q_table)


# ============================================================
# PROGRAM 10: Evaluate trained Q-Learning agent
# ============================================================

import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

evaluation_episodes = 100

successes = 0
total_rewards = 0

for episode in range(evaluation_episodes):

    state, info = env.reset()

    for step in range(100):

        action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        state = next_state

        if terminated or truncated:

            total_rewards += reward

            if reward == 1:
                successes += 1

            break

success_rate = (
    successes / evaluation_episodes
) * 100

print("Evaluation Episodes:", evaluation_episodes)
print("Successful Episodes:", successes)
print("Success Rate:", success_rate)
print("Total Reward:", total_rewards)

env.close()


# ============================================================
# PROGRAM 11: Plot cumulative rewards during training
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

cumulative_rewards = np.cumsum(
    episode_rewards
)

plt.plot(cumulative_rewards)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards During Q-Learning")

plt.show()


# ============================================================
# PROGRAM 12: Effect of different learning rates
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

learning_rates = [
    0.1,
    0.5,
    0.8,
    1.0
]

learning_results = {}

for learning_rate in learning_rates:

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    q_table = np.zeros(
        (
            env.observation_space.n,
            env.action_space.n
        )
    )

    epsilon = 1.0
    rewards = []

    for episode in range(3000):

        state, info = env.reset()
        total_reward = 0

        for step in range(100):

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            q_table[state, action] += learning_rate * (
                reward
                + 0.95 * np.max(q_table[next_state])
                - q_table[state, action]
            )

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)

    learning_results[learning_rate] = rewards

    env.close()

for learning_rate, rewards in learning_results.items():

    moving_average = np.convolve(
        rewards,
        np.ones(100) / 100,
        mode="valid"
    )

    plt.plot(
        moving_average,
        label=f"Learning Rate {learning_rate}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 13: Effect of different Gamma values
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

gamma_values = [
    0.5,
    0.7,
    0.9,
    0.95,
    0.99
]

gamma_results = {}

for gamma in gamma_values:

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    q_table = np.zeros(
        (
            env.observation_space.n,
            env.action_space.n
        )
    )

    epsilon = 1.0
    rewards = []

    for episode in range(3000):

        state, info = env.reset()
        total_reward = 0

        for step in range(100):

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            q_table[state, action] += 0.8 * (
                reward
                + gamma * np.max(q_table[next_state])
                - q_table[state, action]
            )

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)

    gamma_results[gamma] = rewards

    env.close()

for gamma, rewards in gamma_results.items():

    moving_average = np.convolve(
        rewards,
        np.ones(100) / 100,
        mode="valid"
    )

    plt.plot(
        moving_average,
        label=f"Gamma {gamma}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Effect of Discount Factor Gamma")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 14: Compare exploration and exploitation
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

epsilon_values = [
    0.1,
    0.3,
    0.5,
    0.8,
    1.0
]

epsilon_results = {}

for initial_epsilon in epsilon_values:

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    q_table = np.zeros(
        (
            env.observation_space.n,
            env.action_space.n
        )
    )

    epsilon = initial_epsilon
    rewards = []

    for episode in range(2000):

        state, info = env.reset()
        total_reward = 0

        for step in range(100):

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            q_table[state, action] += 0.8 * (
                reward
                + 0.95 * np.max(q_table[next_state])
                - q_table[state, action]
            )

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)

    epsilon_results[initial_epsilon] = rewards

    env.close()

for epsilon, rewards in epsilon_results.items():

    moving_average = np.convolve(
        rewards,
        np.ones(100) / 100,
        mode="valid"
    )

    plt.plot(
        moving_average,
        label=f"Epsilon {epsilon}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Exploration vs Exploitation")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 15: Implement epsilon-greedy action selection
# ============================================================

import numpy as np

def epsilon_greedy_action(
    q_values,
    epsilon,
    action_space
):

    if np.random.random() < epsilon:

        return action_space.sample()

    return np.argmax(q_values)


q_values = np.array([
    0.2,
    0.8,
    0.4,
    0.1
])

epsilon = 0.2

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

selected_action = epsilon_greedy_action(
    q_values,
    epsilon,
    env.action_space
)

print("Selected Action:", selected_action)

env.close()


# ============================================================
# PROGRAM 16: Design a simple Grid World environment
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

class GridWorld:

    def __init__(self, size=5):

        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.state = self.start

        self.actions = {
            0: (-1, 0),   # Up
            1: (1, 0),    # Down
            2: (0, -1),   # Left
            3: (0, 1)     # Right
        }

    def reset(self):

        self.state = self.start

        return self.state

    def step(self, action):

        row, column = self.state

        row_change, column_change = self.actions[action]

        new_row = np.clip(
            row + row_change,
            0,
            self.size - 1
        )

        new_column = np.clip(
            column + column_change,
            0,
            self.size - 1
        )

        self.state = (
            new_row,
            new_column
        )

        if self.state == self.goal:

            reward = 1
            done = True

        else:

            reward = -0.01
            done = False

        return self.state, reward, done


env = GridWorld()

print("Initial State:")
print(env.reset())

state, reward, done = env.step(3)

print("New State:", state)
print("Reward:", reward)
print("Done:", done)


# ============================================================
# PROGRAM 17: Train Q-Learning agent in custom Grid World
# ============================================================

import numpy as np

env = GridWorld()

size = env.size

number_of_states = size * size
number_of_actions = 4

q_table_grid = np.zeros(
    (
        number_of_states,
        number_of_actions
    )
)

learning_rate = 0.8
gamma = 0.95
epsilon = 1.0

episodes = 3000

def state_to_index(state):
    return state[0] * size + state[1]


for episode in range(episodes):

    state = env.reset()

    for step in range(100):

        state_index = state_to_index(state)

        if np.random.random() < epsilon:

            action = np.random.randint(
                number_of_actions
            )

        else:

            action = np.argmax(
                q_table_grid[state_index]
            )

        next_state, reward, done = env.step(
            action
        )

        next_index = state_to_index(
            next_state
        )

        q_table_grid[
            state_index,
            action
        ] += learning_rate * (
            reward
            + gamma * np.max(
                q_table_grid[next_index]
            )
            - q_table_grid[
                state_index,
                action
            ]
        )

        state = next_state

        if done:
            break

    epsilon = max(
        0.01,
        epsilon * 0.995
    )

print("Grid World training completed.")
print(q_table_grid)


# ============================================================
# PROGRAM 18: Visualize optimal path learned by agent
# ============================================================

import matplotlib.pyplot as plt

env = GridWorld()

state = env.reset()

path = [state]

for step in range(100):

    state_index = state_to_index(state)

    action = np.argmax(
        q_table_grid[state_index]
    )

    state, reward, done = env.step(
        action
    )

    path.append(state)

    if done:
        break

path_rows = [
    position[0]
    for position in path
]

path_columns = [
    position[1]
    for position in path
]

plt.plot(
    path_columns,
    path_rows,
    marker="o"
)

plt.gca().invert_yaxis()

plt.xticks(range(env.size))
plt.yticks(range(env.size))

plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Optimal Path in Grid World")

plt.grid()
plt.show()


# ============================================================
# PROGRAM 19: Compare FrozenLake and Grid World performance
# ============================================================

# FrozenLake evaluation

frozen_env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

frozen_success = 0
frozen_episodes = 100

for episode in range(frozen_episodes):

    state, info = frozen_env.reset()

    for step in range(100):

        action = np.argmax(
            q_table[state]
        )

        next_state, reward, terminated, truncated, info = (
            frozen_env.step(action)
        )

        state = next_state

        if terminated or truncated:

            if reward == 1:
                frozen_success += 1

            break

frozen_success_rate = (
    frozen_success / frozen_episodes
) * 100

frozen_env.close()


# Grid World evaluation

grid_success = 0
grid_episodes = 100

for episode in range(grid_episodes):

    state = env.reset()

    for step in range(100):

        state_index = state_to_index(state)

        action = np.argmax(
            q_table_grid[state_index]
        )

        state, reward, done = env.step(
            action
        )

        if done:

            grid_success += 1
            break

grid_success_rate = (
    grid_success / grid_episodes
) * 100

print(
    "FrozenLake Success Rate:",
    frozen_success_rate
)

print(
    "Grid World Success Rate:",
    grid_success_rate
)


# ============================================================
# PROGRAM 20: Analyze convergence of Q-Learning
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

q_table_convergence = np.zeros(
    (
        env.observation_space.n,
        env.action_space.n
    )
)

learning_rate = 0.8
gamma = 0.95
epsilon = 1.0

rewards = []

for episode in range(5000):

    state, info = env.reset()
    total_reward = 0

    for step in range(100):

        if np.random.random() < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(
                q_table_convergence[state]
            )

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        q_table_convergence[
            state,
            action
        ] += learning_rate * (
            reward
            + gamma * np.max(
                q_table_convergence[next_state]
            )
            - q_table_convergence[
                state,
                action
            ]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)

    epsilon = max(
        0.01,
        epsilon * 0.995
    )

env.close()

moving_average = np.convolve(
    rewards,
    np.ones(100) / 100,
    mode="valid"
)

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Convergence")

plt.show()


# ============================================================
# PROGRAM 21: Install required libraries for DQN
# ============================================================

# Run the following commands in Jupyter/Terminal:
#
# !pip install torch
# !pip install gymnasium
# !pip install matplotlib
# !pip install numpy

import torch
import gymnasium as gym

print("DQN libraries configured successfully.")


# ============================================================
# PROGRAM 22: Implement a basic Deep Q-Network using PyTorch
# ============================================================

import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                action_size
            )
        )

    def forward(self, state):

        return self.network(state)


env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

dqn_model = DQN(
    state_size,
    action_size
)

print(dqn_model)

env.close()


# ============================================================
# PROGRAM 23: Train DQN agent on CartPole
# ============================================================

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                state_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                action_size
            )
        )

    def forward(self, state):

        return self.layers(state)


env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_network = DQNNetwork(
    state_size,
    action_size
)

target_network = DQNNetwork(
    state_size,
    action_size
)

target_network.load_state_dict(
    policy_network.state_dict()
)

optimizer = optim.Adam(
    policy_network.parameters(),
    lr=0.001
)

memory = deque(
    maxlen=10000
)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64

episodes = 300

dqn_rewards = []


def choose_action(
    state,
    epsilon
):

    if random.random() < epsilon:

        return env.action_space.sample()

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = policy_network(
            state_tensor
        )

    return q_values.argmax(
        dim=1
    ).item()


def train_network():

    if len(memory) < batch_size:
        return

    batch = random.sample(
        memory,
        batch_size
    )

    states = torch.FloatTensor(
        np.array(
            [item[0] for item in batch]
        )
    )

    actions = torch.LongTensor(
        [item[1] for item in batch]
    ).unsqueeze(1)

    rewards = torch.FloatTensor(
        [item[2] for item in batch]
    )

    next_states = torch.FloatTensor(
        np.array(
            [item[3] for item in batch]
        )
    )

    dones = torch.FloatTensor(
        [item[4] for item in batch]
    )

    current_q_values = policy_network(
        states
    ).gather(
        1,
        actions
    ).squeeze()

    with torch.no_grad():

        next_q_values = target_network(
            next_states
        ).max(
            1
        )[0]

        target_values = rewards + (
            gamma
            * next_q_values
            * (1 - dones)
        )

    loss = nn.MSELoss()(
        current_q_values,
        target_values
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(500):

        action = choose_action(
            state,
            epsilon
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        memory.append(
            (
                state,
                action,
                reward,
                next_state,
                float(done)
            )
        )

        train_network()

        state = next_state

        total_reward += reward

        if done:
            break

    dqn_rewards.append(
        total_reward
    )

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if episode % 10 == 0:

        target_network.load_state_dict(
            policy_network.state_dict()
        )

env.close()

print("DQN training completed.")


# ============================================================
# PROGRAM 24: Plot episode-wise DQN rewards
# ============================================================

import matplotlib.pyplot as plt

plt.plot(
    dqn_rewards
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Episode-wise Rewards")

plt.show()


# ============================================================
# PROGRAM 25: Evaluate trained DQN agent
# ============================================================

import gymnasium as gym
import torch

evaluation_env = gym.make(
    "CartPole-v1"
)

evaluation_episodes = 20

evaluation_rewards = []

for episode in range(
    evaluation_episodes
):

    state, info = evaluation_env.reset()

    total_reward = 0

    for step in range(500):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            action = policy_network(
                state_tensor
            ).argmax(
                dim=1
            ).item()

        next_state, reward, terminated, truncated, info = (
            evaluation_env.step(action)
        )

        total_reward += reward

        state = next_state

        if terminated or truncated:
            break

    evaluation_rewards.append(
        total_reward
    )

print(
    "Average Evaluation Reward:",
    np.mean(evaluation_rewards)
)

evaluation_env.close()


# ============================================================
# PROGRAM 26: Compare Q-Learning and DQN performance
# ============================================================

q_learning_average = np.mean(
    episode_rewards[-100:]
)

dqn_average = np.mean(
    dqn_rewards[-100:]
)

print(
    "Q-Learning Average Reward:",
    q_learning_average
)

print(
    "DQN Average Reward:",
    dqn_average
)

if q_learning_average > dqn_average:

    print(
        "Q-Learning achieved higher "
        "average reward."
    )

elif dqn_average > q_learning_average:

    print(
        "DQN achieved higher "
        "average reward."
    )

else:

    print(
        "Both algorithms achieved "
        "similar average reward."
    )


# ============================================================
# PROGRAM 27: Analyze effect of replay memory on DQN
# ============================================================

import matplotlib.pyplot as plt

memory_sizes = [
    1000,
    5000,
    10000,
    20000
]

memory_results = {}

for memory_size in memory_sizes:

    memory = deque(
        maxlen=memory_size
    )

    rewards = []

    for episode in range(50):

        state, info = env.reset()

        total_reward = 0

        for step in range(200):

            action = choose_action(
                state,
                epsilon
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    float(done)
                )
            )

            state = next_state

            total_reward += reward

            if done:
                break

        rewards.append(
            total_reward
        )

    memory_results[
        memory_size
    ] = rewards

for size, rewards in memory_results.items():

    plt.plot(
        rewards,
        label=f"Memory {size}"
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Effect of Replay Memory")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 28: Study role of target network in DQN
# ============================================================

target_update_intervals = [
    1,
    10,
    50,
    100
]

target_results = {}

for interval in target_update_intervals:

    target_results[interval] = []

    for episode in range(50):

        state, info = env.reset()

        total_reward = 0

        for step in range(200):

            action = choose_action(
                state,
                epsilon
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = next_state

            total_reward += reward

            if terminated or truncated:
                break

        target_results[
            interval
        ].append(total_reward)

        if episode % interval == 0:

            target_network.load_state_dict(
                policy_network.state_dict()
            )

for interval, rewards in target_results.items():

    plt.plot(
        rewards,
        label=f"Update Interval {interval}"
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Effect of Target Network Update")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 29: Save trained DQN model
# ============================================================

torch.save(
    policy_network.state_dict(),
    "dqn_cartpole_model.pth"
)

print(
    "DQN model saved successfully."
)


# ============================================================
# PROGRAM 30: Load saved DQN model and perform testing
# ============================================================

loaded_model = DQNNetwork(
    state_size,
    action_size
)

loaded_model.load_state_dict(
    torch.load(
        "dqn_cartpole_model.pth",
        weights_only=True
    )
)

loaded_model.eval()

test_env = gym.make(
    "CartPole-v1"
)

test_rewards = []

for episode in range(10):

    state, info = test_env.reset()

    total_reward = 0

    for step in range(500):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            action = loaded_model(
                state_tensor
            ).argmax(
                dim=1
            ).item()

        state, reward, terminated, truncated, info = (
            test_env.step(action)
        )

        total_reward += reward

        if terminated or truncated:
            break

    test_rewards.append(
        total_reward
    )

print(
    "Test Rewards:",
    test_rewards
)

test_env.close()


# ============================================================
# PROGRAM 31: Compare cumulative rewards of RL algorithms
# ============================================================

q_cumulative_rewards = np.cumsum(
    episode_rewards
)

dqn_cumulative_rewards = np.cumsum(
    dqn_rewards
)

plt.plot(
    q_cumulative_rewards,
    label="Q-Learning"
)

plt.plot(
    dqn_cumulative_rewards,
    label="DQN"
)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 32: Visualize learning curve of RL agent
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

window_size = 100

q_learning_curve = np.convolve(
    episode_rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

dqn_learning_curve = np.convolve(
    dqn_rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

plt.plot(
    q_learning_curve,
    label="Q-Learning"
)

plt.plot(
    dqn_learning_curve,
    label="DQN"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("RL Learning Curves")

plt.legend()
plt.show()


# ============================================================
# PROGRAM 33: Compare training time and convergence
# ============================================================

import time

# Q-Learning training time

start_time = time.time()

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

q_table_time = np.zeros(
    (
        env.observation_space.n,
        env.action_space.n
    )
)

for episode in range(1000):

    state, info = env.reset()

    for step in range(100):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        q_table_time[
            state,
            action
        ] += 0.8 * (
            reward
            + 0.95 * np.max(
                q_table_time[next_state]
            )
            - q_table_time[
                state,
                action
            ]
        )

        state = next_state

        if terminated or truncated:
            break

q_learning_time = (
    time.time() - start_time
)

env.close()


# DQN training time

start_time = time.time()

env = gym.make(
    "CartPole-v1"
)

for episode in range(100):

    state, info = env.reset()

    for step in range(200):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        state = next_state

        if terminated or truncated:
            break

dqn_training_time = (
    time.time() - start_time
)

env.close()

print(
    "Q-Learning Training Time:",
    q_learning_time
)

print(
    "DQN Training Time:",
    dqn_training_time
)


# ============================================================
# PROGRAM 34: Analyze impact of hyperparameters
# ============================================================

learning_rates = [
    0.1,
    0.5,
    0.8
]

gamma_values = [
    0.8,
    0.95,
    0.99
]

epsilon_values = [
    0.1,
    0.5,
    1.0
]

hyperparameter_results = []

for learning_rate in learning_rates:

    for gamma in gamma_values:

        for epsilon in epsilon_values:

            env = gym.make(
                "FrozenLake-v1",
                is_slippery=False
            )

            q_table_temp = np.zeros(
                (
                    env.observation_space.n,
                    env.action_space.n
                )
            )

            total_reward = 0

            for episode in range(500):

                state, info = env.reset()

                for step in range(100):

                    if (
                        np.random.random()
                        < epsilon
                    ):

                        action = (
                            env.action_space.sample()
                        )

                    else:

                        action = np.argmax(
                            q_table_temp[state]
                        )

                    next_state, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    q_table_temp[
                        state,
                        action
                    ] += learning_rate * (
                        reward
                        + gamma * np.max(
                            q_table_temp[
                                next_state
                            ]
                        )
                        - q_table_temp[
                            state,
                            action
                        ]
                    )

                    state = next_state

                    total_reward += reward

                    if terminated or truncated:
                        break

            average_reward = (
                total_reward / 500
            )

            hyperparameter_results.append(
                {
                    "Learning Rate": learning_rate,
                    "Gamma": gamma,
                    "Epsilon": epsilon,
                    "Average Reward": average_reward
                }
            )

            env.close()

hyperparameter_df = pd.DataFrame(
    hyperparameter_results
)

print(hyperparameter_df)


# ============================================================
# PROGRAM 35: Comparative report of RL algorithms
# ============================================================

import pandas as pd
import numpy as np

q_learning_reward = np.mean(
    episode_rewards[-100:]
)

dqn_reward = np.mean(
    dqn_rewards[-100:]
)

q_learning_time_value = q_learning_time

dqn_time_value = dqn_training_time

comparison_report = pd.DataFrame(
    {
        "Algorithm": [
            "Q-Learning",
            "Deep Q-Network"
        ],

        "Average Final Reward": [
            q_learning_reward,
            dqn_reward
        ],

        "Training Time": [
            q_learning_time_value,
            dqn_time_value
        ]
    }
)

print("REINFORCEMENT LEARNING COMPARATIVE REPORT")
print("=" * 50)

print(comparison_report)

print("\nPerformance Analysis:")

if q_learning_reward > dqn_reward:

    print(
        "Q-Learning achieved a higher "
        "average final reward."
    )

elif dqn_reward > q_learning_reward:

    print(
        "DQN achieved a higher "
        "average final reward."
    )

else:

    print(
        "Both algorithms achieved "
        "similar average rewards."
    )

print("\nObservation:")

print(
    "Q-Learning uses a Q-table and is suitable "
    "for small discrete state and action spaces."
)

print(
    "DQN uses a neural network and is suitable "
    "for larger or continuous state spaces."
)

print("\nConclusion:")

print(
    "Both Q-Learning and DQN are useful "
    "Reinforcement Learning algorithms. "
    "The appropriate algorithm depends on "
    "the environment, state space, action space, "
    "and computational requirements."
)
