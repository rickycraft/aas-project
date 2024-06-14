import collections
import os
import sys

import gym
import keras
import numpy as np
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt

import impala
import model_gym

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
use_impala = os.environ.get("USE_IMPALA", "0") == "1"
done_reward = float(os.environ.get("DONE_REWARD", "0"))
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
suffix = f"{str(lr)}-{str(clip)}"
if use_impala:
    suffix = f"imp-{suffix}"
else:
    suffix = f"nat-{suffix}"
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
huber_loss = keras.losses.Huber(reduction='sum')
optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=clip)
max_episodes = 100_000
replay_size = 500
gamma = 0.999
entropy_weight = 0.01

num_actions = env.action_space.n
obs_space = env.observation_space.shape

if use_impala:
  model = impala.impala_cnn(obs_space, num_actions)
else:
   model = model_gym.build_model_ac(obs_space, num_actions, load_weights=False, init_zero=False)
if load_weights:
    model.load_weights(weights_file)

@tf.numpy_function(Tout=[tf.float32, tf.float32, tf.int32])
def step(action: np.ndarray):
  state, reward, done, info = env.step(action)
  return (state.astype(np.float32),np.array(reward, np.float32),np.array(done, np.int32))

@tf.numpy_function(Tout=[tf.int64])
def choose_action(probs: np.ndarray):
    action = np.random.choice(num_actions, p=np.squeeze(probs))
    return tf.cast(action, tf.int64)

def epoch(initial_state: tf.Tensor, model: keras.Model):
  action_probs = tf.TensorArray(dtype=tf.float32, size=replay_size, dynamic_size=True)
  entropies = tf.TensorArray(dtype=tf.float32, size=replay_size, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=replay_size, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=replay_size, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(1_000_000):
    # Normalize tensor
    state = tf.expand_dims(state / 255, 0)

    # Run the model and to get action probabilities and critic value
    prob, value = model(state)

    # Calculate probability entropy
    entropy = -tf.reduce_sum(prob * tf.math.log(prob + 1e-9), axis=1)

    # Sample next action from the probability distribution
    action = tf.squeeze(choose_action(prob))

    # Apply action to the environment to get next state and reward
    state, reward, done = step(action)
    # Change the reward for the final state (in fruitbot or chaser is useful to punish agent for dying)
    if tf.cast(done, tf.bool):
        reward = done_reward
    state.set_shape(initial_state_shape)

    # Store values a stacked sensor
    action_probs = action_probs.write(t, prob[0, action])
    entropies = entropies.write(t, entropy)
    values = values.write(t, tf.squeeze(value))
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  entropies = entropies.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, entropies, values, rewards

def calc_returns(rewards: tf.Tensor):
    size = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False)
    G = tf.constant(0.0)
    G_shape = G.shape
    n = 0
    for r in tf.cast(rewards[::-1], tf.float32):
        G = r + gamma * G
        G.set_shape(G_shape)
        returns = returns.write(n, G)
        n += 1

    returns = returns.stack()[::-1]
    returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns

@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer) -> tf.Tensor:

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_probs, entropies, values, rewards = epoch(initial_state, model)

    # Calculate the expected returns
    returns = calc_returns(rewards)

    # Convert training data to appropriate TF tensor shapes
    action_probs, entropies, values, returns = [
       tf.expand_dims(x, 1) for x in [action_probs, entropies, values, returns]
    ]

    # Loss calculation
    advantage = tf.stop_gradient(returns - values)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage + entropy_weight * entropies)
    critic_loss = huber_loss(values, returns)
    loss = actor_loss + critic_loss

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)
  return episode_reward

# Keep the last episodes reward
graph_interval = 100
score_logger = collections.deque(maxlen=graph_interval+10)
score_logger_mean = []
score_logger_std = []
total_timestep = 0

t = tqdm.trange(max_episodes)
for episode in t:
    initial_state = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = train_step(initial_state, model, optimizer).numpy()

    score_logger.append(episode_reward)
    running_reward = np.mean(score_logger)
    t.set_postfix(episode_reward=episode_reward, episode=episode, running=running_reward)
    if len(score_logger) > graph_interval:
        score_logger_mean.append(running_reward)
        score_logger_std.append(np.std(score_logger))

    if episode % graph_interval == 0 and episode > 0:
        # print(f"\nEpisode: {episode} Mean: {running_reward} | {suffix}")
        model.save_weights(weights_file)

        plt.clf()
        plt.title(
           f"lr {lr} clip {clip} | start {start_level} | num {num_level} | done {done_reward} | entropy {entropy_weight}"
          )
        x = np.arange(len(score_logger_mean))
        mean_low = np.array(score_logger_mean) - np.array(score_logger_std)
        mean_high = np.array(score_logger_mean) + np.array(score_logger_std)
        plt.plot(x, score_logger_mean, color='blue')
        plt.fill_between(x, mean_low, mean_high, color='cyan', alpha=0.5)
        plt.plot([x[0], x[-1]], [score_logger_mean[0], score_logger_mean[-1]], color='red', linestyle='-', linewidth=1)
        plt.savefig(plt_file)
