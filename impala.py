from keras import Model, layers


def impala_cnn(input_shape, num_actions, advantage=False):
  def residual_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.add([x, inputs])
    x = layers.ReLU()(x)
    return x

  def conv_sequence(x, filters):
    x = layers.Conv2D(filters, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = residual_block(x, filters)
    x = residual_block(x, filters)
    return x

  inputs = layers.Input(shape=input_shape)

  x = conv_sequence(inputs, 16)
  x = conv_sequence(x, 32)
  x = conv_sequence(x, 32)

  x = layers.Flatten()(x)
  x = layers.Dense(256, activation='relu')(x)
  action_logits = layers.Dense(num_actions, activation='softmax')(x)
  if advantage:
    value = layers.Dense(num_actions)(x)
  else:
    value = layers.Dense(1)(x)

  model = Model(inputs=inputs, outputs=[action_logits, value])
  return model