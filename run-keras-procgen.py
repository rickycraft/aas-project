import os
import sys

import gym
import imageio
import numpy as np
import tensorflow as tf

import impala
import nature

start_level = os.environ.get("START_LEVEL")
num_level = int(os.environ.get("NUM_LEVEL", "3"))
env_name = os.environ.get("ENV_NAME", "fruitbot")
use_impala = os.environ.get("USE_IMPALA", "0") == "1"
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
weights_file = sys.argv[1]

env_options = {
  "id": f'procgen:procgen-{env_name}-v0',
  "distribution_mode": "easy",
  "render_mode": "human",
  "rand_seed": seed,
  "num_levels": num_level,
  "use_sequential_levels": False,
  "use_backgrounds": False,
  "restrict_themes": True,
  "use_monochrome_assets": True,
  "use_generated_assets": False
}
if start_level:
  env_options["start_level"] = int(start_level)
print(env_options)
env = gym.make(**env_options)

num_actions = env.action_space.n
obs_space = env.observation_space.shape
print(f"num_actions: {num_actions}")
print(f"obs_space: {obs_space}")

if use_impala:
  model = impala.impala_cnn(obs_space, num_actions)
else:
  model = nature.build_model_ac(obs_space, num_actions, load_weights=False, init_zero=False)
model.load_weights(weights_file)
print(f'Loaded weights {weights_file}')

# Extract info from the weights file
# good/chaser-imp-0.0001-42.0.weights.h5
lr = weights_file.split('-')[2]
clip = weights_file.split('-')[3][:-11]
imp_str = 'imp' if use_impala else 'nat'
gif_base = f'gif/{env_name}-{imp_str}-{lr}-{clip}'
print('gif', gif_base)

for episode in range(100):
  state = env.reset()
  state = tf.expand_dims(state / 255, 0)
  rewards = []
  frames = []

  while True:
    frames.append(env.render(mode='rgb_array'))
    action_probs, _ = model(state)
    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
    state, reward, done, _ = env.step(action)
    state = tf.expand_dims(state / 255, 0)
    rewards.append(reward)

    if done:
      break

  imageio.mimsave(f'{gif_base}-{episode}.gif', frames, format='gif', fps=15)
  frames = []

  print(f"{episode}\t| Total reward: {np.sum(rewards)}")
