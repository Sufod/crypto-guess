import os
import pandas as pd
import csv
import numpy as np

from misc.Logger import Logger
from misc.Utils import Utils
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

    @staticmethod
    def produce_train_dev_test_from_full_corpus(filepath):
        basename, dir_name = Utils.get_base_and_dir_names(filepath)

        corpus = DataLoader.load_csv_data(filepath)
        train, dev, test = np.split(corpus, [int(.8 * len(corpus)), int(.9 * len(corpus))])
        sub_corpora = {
            'train': train,
            'dev': dev,
            'test': test
        }
        for sub_corpus_name, sub_corpus in sub_corpora.items():
            sub_corpus_path = Utils.get_sub_corpus_path(basename, dir_name, sub_corpus_name)
            sub_corpus.to_csv(path_or_buf=sub_corpus_path, index=False)
            Logger.okgreen("Created sub corpus : " + sub_corpus_path)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        Logger.bold("=== ---------- ===")

    def __init__(self, params):
        self.params = params
        Logger.bold("=== DataLoader ===")
        autogen_path = params["corpora"].pop('autogen', None)
        if autogen_path is not None:
            Logger.header("--- Autogen sequence ---")
            basename, dir_name = Utils.get_base_and_dir_names(autogen_path)
            time_diff = os.path.getmtime(autogen_path) - os.path.getmtime(
                Utils.get_sub_corpus_path(basename, dir_name, "train"))
            if time_diff > 200.0:
                Logger.header("-- Creating sub corpora from :" + autogen_path + " --")
                DataLoader.produce_train_dev_test_from_full_corpus(autogen_path)
            else:
                Logger.okgreen(" All corpora are up-to-date with " + autogen_path)

            for sub_corpus in ["train", "dev", "test"]:
                params["corpora"][sub_corpus] = Utils.get_sub_corpus_path(basename, dir_name, sub_corpus)

        Logger.bold("== Reading Params ==")
        for sub_corpus in ["train", "dev", "test"]:
            Logger.okblue(
                "Using " + sub_corpus + " corpus : " + Utils.get_sub_corpus_path(basename, dir_name, sub_corpus))
        Logger.bold("== Loading corpora ==")

    def load_test_dataframes(self):
        filename = self.params["corpora"]["test"]
        return self.load_dataframes(filename)

    def load_dev_dataframes(self):
        filename = self.params["corpora"]["dev"]
        return self.load_dataframes(filename)

    def load_train_dataframes(self):
        filename = self.params["corpora"]["train"]
        return self.load_dataframes(filename)

    def load_dataframes(self, filename):
        corpus = DataLoader.load_csv_data(filename)
        data_converter = DataConverter(corpus)
        features, labels = data_converter.generate_features_and_labels(self.params)
        Logger.okgreen("Loaded corpus : " + filename)
        return features, labels
