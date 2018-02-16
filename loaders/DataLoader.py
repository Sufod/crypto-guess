import tensorflow as tf
import pandas as pd
import csv

from processors.DataConverter import DataConverter


class DataLoader:
    TRAIN_URL = "http://crypto_training.csv"
    TEST_URL = "http://crypto_training.csv"

    CSV_COLUMN_NAMES = []

    def maybe_download(self):
        train_path = tf.keras.utils.get_file(self.TRAIN_URL.split('/')[-1], self.TRAIN_URL)
        test_path = tf.keras.utils.get_file(self.TEST_URL.split('/')[-1], self.TEST_URL)
        return train_path, test_path

    def load_csv_data(self, filepath):
        # Create a dataset containing the text lines.
        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            self.CSV_COLUMN_NAMES = next(reader)

        dataset = pd.read_csv(filepath, names=self.CSV_COLUMN_NAMES, header=0)
        return dataset

    def load_test_dataframes(self, params):
        filename = params["corpora"]["test_corpus"]
        return self.load_dataframes(filename, params)

    def load_dev_dataframes(self, params):
        filename = params["corpora"]["dev_corpus"]
        return self.load_dataframes(filename, params)

    def load_train_dataframes(self, params):
        filename = params["corpora"]["train_corpus"]
        return self.load_dataframes(filename, params)

    def load_dataframes(self, filename, params):
        corpus = self.data_loader.load_csv_data(filename)
        test_data_converter = DataConverter(corpus)
        test_features, test_labels = test_data_converter.generate_features_and_labels(params)
        return test_features, test_labels
