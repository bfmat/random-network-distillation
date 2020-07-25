import collections
import time
import cv2
import gym
import minerl
import numpy as np

class Minecraft:

  ACTION_KEYS_CHOP = ('attack', 'back', 'camera', 'forward', 'jump', 'left', 'right')
  ACTION_SET_CHOP = (
      (1, 0, (0, 0), 0, 0, 0, 0),    # Attack
      (0, 1, (0, 0), 0, 0, 0, 0),    # Back
      (0, 0, (-10, 0), 0, 0, 0, 0),  # Look Left
      (0, 0, (10, 0), 0, 0, 0, 0),   # Look Right
      (0, 0, (0, -10), 0, 0, 0, 0),  # Look Down
      (0, 0, (0, 10), 0, 0, 0, 0),   # Look Up
      (0, 0, (0, 0), 1, 0, 0, 0),    # Forward
      (0, 0, (0, 0), 1, 1, 0, 0),    # Jump + Forward
      (0, 0, (0, 0), 0, 0, 1, 0),    # Left
      (0, 0, (0, 0), 0, 0, 0, 1),    # Right
  )

  ACTION_KEYS_NAVIGATE = ('attack', 'back', 'camera', 'forward', 'jump', 'left', 'place', 'right')
  ACTION_SET_NAVIGATE = (
      (1, 0, (0, 0), 0, 0, 0, 0, 0),    # Attack
      (0, 1, (0, 0), 0, 0, 0, 0, 0),    # Back
      (0, 0, (-10, 0), 0, 0, 0, 0, 0),  # Look Left
      (0, 0, (10, 0), 0, 0, 0, 0, 0),   # Look Right
      (0, 0, (0, -10), 0, 0, 0, 0, 0),  # Look Down
      (0, 0, (0, 5), 0, 0, 0, 0, 0),   # Look Up
      (0, 0, (0, 0), 1, 0, 0, 0, 0),    # Forward
      (0, 0, (0, 0), 1, 1, 0, 0, 0),    # Jump + Forward
      (0, 0, (0, 0), 0, 0, 1, 0, 0),    # Left
      (0, 0, (0, 0), 0, 0, 0, 1, 0),    # Place
      (0, 0, (0, 0), 0, 0, 0, 0, 1),    # Right
  )

  ACTION_SET_KEYS = {'MineRLTreechop-v0': (ACTION_KEYS_CHOP, ACTION_SET_CHOP),
                     'MineRLNavigateDense-v0': (ACTION_KEYS_NAVIGATE, ACTION_SET_NAVIGATE), 
                     'MineRLNavigate-v0': (ACTION_KEYS_NAVIGATE, ACTION_SET_NAVIGATE), 
                     'MineRLNavigateExtremeDense-v0': (ACTION_KEYS_NAVIGATE, ACTION_SET_NAVIGATE), 
                     'MineRLNavigateExtreme-v0': (ACTION_KEYS_NAVIGATE, ACTION_SET_NAVIGATE),
                     'MineRLObtainIronPickaxe-v0': None, 
                     'MineRLObtainIronPickaxeDense-v0': None, 
                     'MineRLObtainDiamond-v0': None, 
                     'MineRLObtainDiamondDense-v0': None}

  def __init__(
      self, task, mode, size=(84, 84), action_repeat=1, 
      buffer_size=1024, seed=0, attack_repeat=15, action_set_keys=ACTION_SET_KEYS):
    print(f"{mode}: Creating Environment... ")
    start = time.time()
    self._env = gym.make(task)
    self._env.seed(seed)
    print(f"{mode}: {task} Created: {time.time()-start:.2f}s")
    
    self._task = task
    self._mode = mode
    self._size = size
    self._seed = seed

    self._action_repeat = action_repeat
    self._attack_repeat = attack_repeat
    self._action_keys, self._action_set = action_set_keys[task][0], action_set_keys[task][1]

    self._camera_angle = np.array([0.0, 0.0])
    self._camera_clip = 60.0

    self.metadata = {}
    self.spec = None

  def _mc_action(self, raw_action):
    # Mapping raw action to Minecraft action
    mc_actions = self._action_set[raw_action]
    act_dict = collections.OrderedDict()
    for k, v in zip(self._action_keys, mc_actions):
      if k == 'camera': # clip the camera angle
        v = np.array(v, dtype=np.float32)
        new_camera_angle = np.clip(self._camera_angle + v, -self._camera_clip, self._camera_clip)
        cond = (np.abs(self._camera_angle + v) >= self._camera_clip)
        if cond.any():
          v[cond] = (np.sign(v) * (self._camera_clip - np.abs(self._camera_angle)))[cond]
        self._camera_angle = new_camera_angle
      act_dict[k] = v
    return act_dict

  def _vec_obs(self, mc_obs):
    return cv2.resize(mc_obs['pov'], self._size, interpolation=cv2.INTER_AREA)

  @property
  def observation_space(self):
    return gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(self._action_set))

  def close(self):
    self._env.close()

  def reset(self):
    obs = self._vec_obs(self._env.reset())
    self._camera_angle = np.array([0.0, 0.0])
    return obs

  def step(self, action):
    if not isinstance(action, collections.OrderedDict):
      action = self._mc_action(action)
    repeats = self._attack_repeat if action['attack'] == 1 else self._action_repeat
    repeats = 1 if 'place' in action and action['place'] > 0 else repeats
    rewards = 0
    for _ in range(repeats):
      obs, reward, done, info = self._env.step(action)
      rewards += reward
      if done:
        break
    obs = self._vec_obs(obs)
    return obs, rewards, done, info

  def render(self):
    raise NotImplementedError("Minecraft's Rendering Not Implemented")
