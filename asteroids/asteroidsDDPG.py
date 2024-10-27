import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from asteroidsEnv import AsteroidsEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from train_model import train_model
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Initialize environment
env = AsteroidsEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Set hyperparameters
num_episodes = 10000
max_steps_per_episode = 2500
num_training_steps = 128
batch_size = 128
discount = 0.95
tau = 0.005

# Define actor and critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.max_action * torch.tanh(self.layer3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        action = action.squeeze(1)
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Initialize actor and critic
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

driver_actor = Actor(state_dim, action_dim, max_action).to(device)
driver_critic = Critic(state_dim, action_dim).to(device)
target_driver_actor = Actor(state_dim, action_dim, max_action).to(device)
target_driver_critic = Critic(state_dim, action_dim).to(device)

# Load actor and critic to target networks
target_driver_actor.load_state_dict(driver_actor.state_dict())
target_driver_critic.load_state_dict(driver_critic.state_dict())

# Set optimizers for actor and critic
driver_actor_optimizer = optim.AdamW(driver_actor.parameters())
driver_critic_optimizer = optim.AdamW(driver_critic.parameters())

# Define replay buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones)

# Initialize replay buffer
driver_replay_buffer = ReplayBuffer()

def plot_rewards(episode_rewards_driver, num_episodes, show_result=False):
    plt.figure(2)
    rewards_t_driver = torch.tensor(episode_rewards_driver, dtype=torch.float)
    if show_result:
        plt.title(f'Episode {num_episodes}: Driver: {episode_rewards_driver[-1]}; Average: {round(np.average(episode_rewards_driver))}; Last 100: {round(np.average(episode_rewards_driver[-100:])) if len(episode_rewards_driver) > 100 else round(np.average(episode_rewards_driver))}')
    else:
        plt.clf()
        plt.title(f'Episode {num_episodes}: Driver: {episode_rewards_driver[-1]}; Average: {round(np.average(episode_rewards_driver))}; Last 100: {round(np.average(episode_rewards_driver[-100:])) if len(episode_rewards_driver) > 100 else round(np.average(episode_rewards_driver))}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.plot(rewards_t_driver.numpy(), color='green')
    # Take 100 episode averages and plot them too
    if len(rewards_t_driver) >= 100:
        means_driver = rewards_t_driver.unfold(0, 100, 1).mean(1).view(-1)
        means_driver = torch.cat((torch.zeros(99), means_driver))
        plt.plot(means_driver.numpy())
        plt.axvline(x=len(episode_rewards_driver) - 100, color='lightgreen', linestyle='--', linewidth=1)
    plt.axhline(y= np.mean(episode_rewards_driver), color='lightgreen', linestyle='--', linewidth=1) if len(episode_rewards_driver) < 100 else plt.axhline(y= round(np.average(episode_rewards_driver[-100:])), color="lightgreen", linestyle='--', linewidth=1)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Main training loop
def series(series_len, played_episodes):
    try:
        driver_actor.load_state_dict(torch.load(f'saves/driver/actor_episode_{played_episodes}.pth'))
        with open('saves/episode_rewards.csv', 'r') as f:
            episode_rewards_driver = []
            for rewards in f.readlines():
                episode_rewards_driver.append(float(rewards.strip().split(',')[0][1:]))
        print("Models and episode rewards loaded successfully.")
    except FileNotFoundError:
        print("Saved models and/or episode rewards not found. Starting from scratch.")
        episode_rewards_driver = []

    for episode in tqdm(range(series_len), desc="Training Progress"):
        average_reward_driver = np.average(episode_rewards_driver[-100:]) if len(episode_rewards_driver) > 100 else np.average(episode_rewards_driver)
    
        if episode % 10 == 0:
            print(f"Episode {played_episodes + episode} of {num_episodes}")

        if episode % 100 == 0 and episode != 0:
            # Save the latest models and episode rewards
            torch.save(driver_actor.state_dict(), f'saves/driver/actor_episode_latest.pth')
            torch.save(driver_critic.state_dict(), f'saves/driver/critic_episode_latest.pth')
            with open('saves/episode_rewards_latest.csv', 'w') as f:
                for i in range(len(episode_rewards_driver)):
                    f.write(f"{episode_rewards_driver[i]}\n")
            print(f"Saved latest models and episode rewards at episode {played_episodes + episode}")

        state = env.reset()
        episode_reward_driver = 0
        driver_done = False

        for step in range(max_steps_per_episode):
            driver_action = driver_actor(state.to(device)).cpu().data.numpy() if not driver_done else np.zeros(action_dim)
            next_state, rewards, dones, _ = env.step(driver_action)

            driver_replay_buffer.add((state, next_state, driver_action, rewards, dones)) if not driver_done else None

            state = next_state
            episode_reward_driver += rewards

            if step == max_steps_per_episode - 1 or dones:
                driver_done = True

            if driver_done:
                episode_rewards_driver.append(episode_reward_driver)
                plot_rewards(episode_rewards_driver, played_episodes + episode)
                break


        # Train actors and critics after each episode
        train_model(num_training_steps, batch_size, discount, tau, device, driver_replay_buffer, driver_actor, driver_critic, target_driver_actor, target_driver_critic, driver_actor_optimizer, driver_critic_optimizer)


    # save the models
    torch.save(driver_actor.state_dict(), f'saves/driver/actor_episode_{played_episodes + series_len}.pth')
    torch.save(driver_critic.state_dict(), f'saves/driver/critic_episode_{played_episodes + series_len}.pth')
    # save csv of episode_rewards
    with open('saves/episode_rewards.csv', 'w') as f:
        for i in range(len(episode_rewards_driver)):
            f.write(f'{episode_rewards_driver[i]}\n')

def main(starting_episode=0):
    series_len = 1000
    if starting_episode > 0:
        played_episodes = starting_episode
    else:
        played_episodes = 0
    for played_episodes in tqdm(range(played_episodes, num_episodes, series_len), desc="Training Progress"):
        series(series_len, played_episodes)
        played_episodes += series_len
        print(f'Played {played_episodes}/{num_episodes} episodes.')
    print('Training complete.')

# Train the model
main(0)
with open('saves/episode_rewards.csv', 'r') as f:
    episode_rewards_driver = [float(reward.strip()) for reward in f.readlines()]
# plot the rewards
plot_rewards(episode_rewards_driver, 0, show_result=True)
plt.ioff()
plt.show()
