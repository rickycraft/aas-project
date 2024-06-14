import collections
import os
import sys

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import impala

os.environ["KERAS_BACKEND"] = "tensorflow"
if len(sys.argv) < 2:
  print(f"Usage: {sys.argv[0]} <lr> [clip]")
  sys.exit(1)

lr = float(sys.argv[1])
assert 0 < lr < 1e-2

if len(sys.argv) == 3:
  clip = float(sys.argv[2])
  assert 0.01 < clip
else:
  clip = None
print(f"lr: {lr} clip: {clip}")

start_level = os.environ.get("START_LEVEL")
num_level = int(os.environ.get("NUM_LEVEL", "3"))
init_zero = os.environ.get("INIT_ZERO", "0") == "1"
env_name = os.environ.get("ENV_NAME", "fruitbot")
load_weights = os.environ.get("LOAD_WEIGHTS", "0") == "1"
done_reward = float(os.environ.get("DONE_REWARD", "0"))
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
suffix = f"ad-{str(lr)}-{str(clip)}"
plt_file = f'{env_name}-{suffix}.png'
weights_file = f'{env_name}-{suffix}.weights.h5'

env_options = {
    "id": f'procgen:procgen-{env_name}-v0',
    "distribution_mode": "easy",
    "render_mode": "rgb_array",
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

eps = np.finfo(np.float32).eps.item()
huber_loss = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=clip)
max_episodes = 100_000
replay_size = 500
gamma = 0.999
entropy_weight = 0.01

num_actions = env.action_space.n
obs_space = env.observation_space.shape

model = impala.impala_cnn(obs_space, num_actions, advantage=True)
if load_weights:
  model.load_weights(weights_file)

@tf.numpy_function(Tout=[tf.float32, tf.float32, tf.int32])
def step(action: np.ndarray):
  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32),np.array(reward, np.float32),np.array(done, np.int32))

def normalize_state(state: tf.Tensor):
  tensor = tf.convert_to_tensor(state, dtype=tf.float32)
  return tf.expand_dims(tensor / 255, 0)

@tf.numpy_function(Tout=[tf.int64])
def choose_action(probs: np.ndarray):
  action = np.random.choice(num_actions, p=np.squeeze(probs))
  return tf.cast(action, tf.int64)

@tf.function
def train(initial_observation: tf.Tensor):
  initial_shape = initial_observation.shape
  with tf.GradientTape() as tape:
    prob, value = model(initial_observation)

    action = tf.squeeze(choose_action(prob))
    log_prob = tf.math.log(prob[0, action])
    entropy = -tf.reduce_sum(prob * tf.math.log(prob + 1e-9), axis=1)

    observation, reward, done = step(action)
    if tf.cast(done, tf.bool):
      reward = done_reward
    observation = normalize_state(observation)
    observation.set_shape(initial_shape)

    _, next_value = model(observation)

    q_value = value[0, action]
    q_max = tf.argmax(next_value[0])
    q_next = reward + gamma * next_value[0, q_max]
    advantage = tf.stop_gradient(q_next - q_value)
    action_loss = -1 * (log_prob * advantage + entropy_weight * entropy)
    critic_loss = tf.square(q_value - q_next)
    agent_loss = action_loss + critic_loss

    grads = tape.gradient(agent_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return reward, observation, tf.cast(done, tf.bool)

graph_interval = 100
score_logger = collections.deque(maxlen=graph_interval+10)
score_logger_mean = []
score_logger_std = []
total_timestep = 0
t = tqdm()

for episode in range(max_episodes):
  state = env.reset()
  state = normalize_state(state)
  rewards = []
  done = False

  for timestep in range(1_000_000):
    reward, state, done = train(state)
    rewards.append(reward)
    t.update(1)
    if done:
      break

  total_timestep += timestep
  episode_reward = float(tf.reduce_sum(rewards))
  score_logger.append(episode_reward)
  running_reward = np.mean(score_logger)
  t.set_postfix(episode_reward=episode_reward, episode=episode, running=running_reward)
  if len(score_logger) > graph_interval:
    score_logger_mean.append(running_reward)
    score_logger_std.append(np.std(score_logger))

  if episode % graph_interval == 0 and episode > 0:
    print(f"\nEpisode: {episode} Mean: {running_reward} | {suffix}")
    model.save_weights(weights_file)

    plt.clf()
    frame = round(total_timestep / 1e6, 2)
    plt.title(f"lr {lr} clip {clip} | start {start_level} | num {num_level} | entropy {entropy_weight} | f {frame}M")
    x = np.arange(len(score_logger_mean))
    mean_low = np.array(score_logger_mean) - np.array(score_logger_std)
    mean_high = np.array(score_logger_mean) + np.array(score_logger_std)
    plt.plot(x, score_logger_mean, color='blue')
    plt.fill_between(x, mean_low, mean_high, color='cyan', alpha=0.5)
    plt.plot([x[0], x[-1]], [score_logger_mean[0], score_logger_mean[-1]], color='red', linestyle='-', linewidth=1)
    plt.savefig(plt_file)
    # show distribution of probs
    prob, value = model(state)
    print(value[0, 0].numpy(), np.squeeze(prob))
    print('entropy', -tf.reduce_sum(prob * tf.math.log(prob + 1e-9), axis=1).numpy())
