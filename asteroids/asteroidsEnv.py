import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from asteroids import Asteroids

def normalize(value, min_value, max_value):
    return 2 * ((value - min_value) / (max_value - min_value)) - 1

class AsteroidsEnv(gym.Env):
    def __init__(self):
        super(AsteroidsEnv, self).__init__()
        self.game = None
        self.player = None

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        low = np.zeros(109)
        high = np.ones(109)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.frames_x = 400
        self.frames_y = 400

    def reset(self):
        self.game = Asteroids(self.frames_x, self.frames_y)
        return torch.FloatTensor(self.get_observations())

    def step(self, action):
        self.game.player.handle_input(action)
        self.game.update()
        observations = self.get_observations()
        rewards = self.get_reward()
        dones = self.get_done()
        infos = self.get_info()
        self.game.clock.tick(60)
        return torch.FloatTensor(observations), rewards, dones, infos


    def get_observations(self):
        observations = [
            normalize(self.game.player.x, 0, self.frames_x),
            normalize(self.game.player.y, 0, self.frames_y),
            normalize(self.game.player.velocity_vector.x, -10, 10),
            normalize(self.game.player.velocity_vector.y, -10, 10),
            normalize(self.game.player.thrust_vector.x, -1, 1),
            normalize(self.game.player.thrust_vector.y, -1, 1),
            normalize(self.game.player.thrust_vector.magnitude, 0, 10),
            normalize(self.game.player.thrust_vector.angle, -np.pi, np.pi),
            normalize(self.game.player.health / self.game.player.max_health, 0, 1)
        ] # 9 observations
        for asteroid in self.game.asteroids: # 5 observations per asteroid
            observations.append(normalize(asteroid.x, 0, self.frames_x))
            observations.append(normalize(asteroid.y, 0, self.frames_y))
            observations.append(normalize(asteroid.velocity_vector.x, -10, 10))
            observations.append(normalize(asteroid.velocity_vector.y, -10, 10))
            observations.append(normalize(asteroid.size, 0, 20))

        # fill with 0s
        observations.extend([0] * (109 - len(observations)))
        return observations
    
    def get_reward(self):
        reward = 0
        reward += self.game.player.health / self.game.player.max_health
        reward += self.game.max_asteroids_counter
        if self.game.player.health <= 0:
            reward -= 100
        return reward
    
    def get_done(self):
        return self.game.player.health <= 0
    
    def get_info(self):
        return {}