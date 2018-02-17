import pandas as pd
import csv

from processors.DataConverter import DataConverter


class DataLoader:

    @staticmethod
    def load_csv_data(filepath):
        # Create a dataset containing the text lines.
        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            CSV_COLUMN_NAMES = next(reader)

        dataset = pd.read_csv(filepath, names=CSV_COLUMN_NAMES, header=0)
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
        corpus = DataLoader.load_csv_data(filename)
        test_data_converter = DataConverter(corpus)
        test_features, test_labels = test_data_converter.generate_features_and_labels(params)
        return test_features, test_labels
