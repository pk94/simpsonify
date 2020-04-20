import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, label, metafile_path='metafile.csv'):
        self.metafile_path = metafile_path
        self.metafile = pd.DataFrame()
        self.label = label
        self.data_len = {}

    def load_and_shuffle_metafile(self):
        metafile = pd.read_csv(self.metafile_path)
        metafile = metafile.sample(frac=1).reset_index(drop=True)
        self.metafile = metafile.loc[metafile['label'] == self.label]
        self.data_len = int(self.metafile.count()['label'])

    def data_generator(self):
        for idx, row in self.metafile.iterrows():
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
        dataset = tf.data.Dataset.from_generator(self.data_generator, (tf.string), (tf.TensorShape(())))
        dataset = dataset.shuffle(self.data_len)
        dataset = dataset.map(lambda image: tuple(tf.py_function(self.process_data, [image], [tf.float32])),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.take(9800).batch(1)
        return dataset


