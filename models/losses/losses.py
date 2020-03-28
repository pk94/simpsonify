from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf

def discriminator_loss(real, generated):
    loss_obj = BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, lbd = 10):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return lbd * loss1

def identity_loss(real_image, same_image, lbd = 10):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return lbd * 0.5 * loss