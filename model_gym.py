import keras
from keras import layers

zero = keras.initializers.Zeros()
base_weights = 'files/ac-base.weights.h5'

def build_model_ac(input_shape, num_actions, init_zero = False, load_weights = True):
  inputs = layers.Input(shape=input_shape)
  x = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
  x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
  x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation="relu")(x)

  if init_zero:
    actor = layers.Dense(num_actions,kernel_initializer=zero, activation="softmax")(x)
    critic = layers.Dense(1, kernel_initializer=zero)(x)
  else:
    actor = layers.Dense(num_actions, activation="softmax")(x)
    critic = layers.Dense(1)(x)

  model = keras.Model(inputs=inputs, outputs=[actor, critic])
  print(model.summary())
  if load_weights:
    model.load_weights(base_weights)
