import tensorflow as tf
import pandas as pd
import csv


class CryptoDataLoader:
    TRAIN_URL = "http://crypto_training.csv"
    TEST_URL = "http://crypto_training.csv"

    CSV_COLUMN_NAMES = []
    DEFAULTS_FIELDS = [[0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    def maybe_download(self):
        train_path = tf.keras.utils.get_file(self.TRAIN_URL.split('/')[-1], self.TRAIN_URL)
        test_path = tf.keras.utils.get_file(self.TEST_URL.split('/')[-1], self.TEST_URL)
        return train_path, test_path

    def load_crypto_crawler_data(self, filepath):
        # Create a dataset containing the text lines.
        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            self.CSV_COLUMN_NAMES = next(reader)

        dataset = pd.read_csv(filepath, names=self.CSV_COLUMN_NAMES, header=0)
        return dataset
