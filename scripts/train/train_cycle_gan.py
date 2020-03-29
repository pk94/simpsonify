from models.architectures.discriminators import Discriminator
from models.architectures.generators import Generator
from models.losses.losses import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from scripts.data.load_dataset import DataLoader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

def train_step(real_x, real_y, generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

def show_image(img, name):
    img = tf.squeeze(img, axis=0)
    plt.imshow(img * 0.5 + 0.5)
    plt.savefig(name)
    plt.clf()

def generate_images(model, test_input_path):
    image = cv2.imread(test_input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    prediction = model(image)
    plt.figure(figsize=(12, 12))
    display_list = [image[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('fig.png')
    plt.clf()
    # plt.show()

def train_loop():
    data_loader_simpson = DataLoader(label='simpson',
                             metafile_path=f'C:\\Users\\kowal\\PycharmProjects\\colorize_gan\\scripts\\data\\metafile.csv')
    datset_simpson = data_loader_simpson.load_dataset()
    data_loader_human = DataLoader(label='human',
                             metafile_path=f'C:\\Users\\kowal\\PycharmProjects\\colorize_gan\\scripts\\data\\metafile.csv')
    datset_human = data_loader_human.load_dataset()
    generator_g = Generator()
    generator_f = Generator()
    discriminator_x = Discriminator()
    discriminator_y = Discriminator()

    generator_g_optimizer = Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_x_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)

    # for image_x, image_y in tf.data.Dataset.zip((datset_human, datset_simpson)):
    #     print('Saving')
    #     show_image(image_x, 'human.png')
    #     show_image(image_y, 'simpson.png')

    for epoch in range(50):
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((datset_human, datset_simpson)):
            print(n)
            train_step(image_x[0], image_y[0], generator_g, generator_f, discriminator_x, discriminator_y,
                       generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer,
                       discriminator_y_optimizer)
            generate_images(generator_g, 'zdjecie.jpg')
            if n % 10 == 0:
                print('.', end='')
            show_image(image_x[0], 'human.png')
            show_image(image_y[0], 'simpson.png')
            n += 1


def main():
    train_loop()


main()
