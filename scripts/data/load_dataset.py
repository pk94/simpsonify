import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

class DataLoader():
    def __init__(self, metafile_path='metafile.csv'):
        self.metafile_path = metafile_path
        self.metafile = pd.DataFrame()
        self.unique_labels = []
        self.labels_encoding = {}
        self.data_len = {}

    def load_and_shuffle_metafile(self):
        metafile = pd.read_csv(self.metafile_path)
        metafile = metafile.sample(frac=1).reset_index(drop=True)
        self.unique_labels = metafile['label'].unique()
        for idx, label in enumerate(self.unique_labels):
            self.labels_encoding.update({label: idx})
        self.metafile = metafile
        for label in self.unique_labels:
            self.data_len.update({label: int(metafile.loc[metafile['label'] == label].count()['label'])})

    def data_generator(self, label):
        for idx, row in self.metafile.iterrows():
            if row['label'] == label:
                yield row['path']

    def process_data(self, filepath):
        filepath = str(filepath.numpy()).replace('\\\\', '\\')[2:-1]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = cv2.normalize(image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return image

    def load_dataset(self):
        self.load_and_shuffle_metafile()
        train_datasets = {}
        for label in self.unique_labels:
            dataset = tf.data.Dataset.from_generator(lambda: self.data_generator(label=label),
                                                     (tf.string), (tf.TensorShape(())))
            dataset = dataset.shuffle(self.data_len[label])
            dataset = dataset.map(lambda image: tuple(tf.py_function(self.process_data, [image], [tf.float32])),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_datasets.update({label: dataset.take(9800).batch(1)})
        return train_datasets


