import gym, numpy as np

# This wrapper only shrinks the state space, eliminating variables that are
# always 0. By definition it does not affect the RL in any way except by making
# network forward/backward prop faster.
# Again, because the state vars omitted by this wrapper are always 0, the RL
# task itself is neither harder nor easier. Just cheaper to run.
# These are the indices of state variables actually used by OpenAI gym Humanoid:
INDS=[  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 57, 58, 59, 60, 61, 62, 63,
       65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99,100,101,102,103,
      105,106,107,108,109,110,111,112,113,115,116,117,118,119,120,121,122,123,
      125,126,127,128,129,130,131,132,133,135,136,137,138,139,140,141,142,143,
      145,146,147,148,149,150,151,152,153,155,156,157,158,159,160,161,162,163,
      165,166,167,168,169,170,171,172,173,175,176,177,178,179,180,181,182,183,
      191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,
      209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,
      227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,
      245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,
      263,264,265,266,267,268,275,276,277,278,279,280,281,282,283,284,285,286,
      287,288,289,290,291]

class HumanoidWrapper():
  def __init__(self, comm, task):
    self.env = gym.make(task)
    assert(self.env.observation_space.shape[0] == 376)
    assert(len(self.env.observation_space.shape) == 1)

    dimAction = self.action_space.shape[0]
    dimState = 257 # SOME UNUSED STATE VARIABLES ARE CUT OUT FOR SPEED
    comm.set_state_action_dims(dimState, dimAction, 0) # 1 agent
    comm.set_action_scales(self.action_space.high, self.action_space.low, False, 0)

  def reset(self):
    observation = self.env.reset()
    return observation[INDS]

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def _max_episode_steps(self):
    return self.env._max_episode_steps

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation[INDS], reward, done, info

  def render(self, mode):
    return self.env.render(mode=mode)

  def close(self):
    return self.env.close()

