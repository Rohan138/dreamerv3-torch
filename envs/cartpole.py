import gym
import numpy as np

class CartPoleEnv(gym.Env):
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            }
        )

    @property
    def action_space(self):
        return self.env.action_space
    
    def _wrap_obs(self, obs, terminated, truncated, is_first):
        import cv2
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_LINEAR)
        obs = {"image": obs}
        obs["is_last"] = terminated or truncated
        obs["is_terminal"] = terminated
        obs["is_first"] = is_first
        return obs

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = action.argmax(-1)
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self.env.render()
        done = terminated or truncated
        obs = self._wrap_obs(obs, terminated, truncated, False)
        return obs, reward, done, info

    def reset(self):
        _, _ = self.env.reset()
        obs = self.env.render()
        obs = self._wrap_obs(obs, False, False, True)
        return obs