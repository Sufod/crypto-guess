import tensorflow as tf
import pandas as pd

from controllers.CryptoFeaturesPreprocessor import CryptoFeaturesPreprocessor
from extractors.CryptoFeaturesExtractor import CryptoFeaturesExtractor

from controllers.CryptoUtils import CryptoUtils


class CryptoDataConverter:
    features_preprocessor = CryptoFeaturesPreprocessor()
    features_extractor = CryptoFeaturesExtractor()

    def __init__(self, corpus):
        self.corpus = corpus

    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), dict(labels)))

        # Shuffle, repeat, and batch the examples.
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    def eval_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, dict(labels))

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset

    def perform(self, fun, *args):
        test = fun(*args)
        return fun(*args)

    def generate_features_and_labels(self, params):
        self.features_preprocessor.preprocess_features(self.corpus)

        labels = pd.DataFrame()
        for task_name, task in params["tasks"].items():
            labels = pd.concat([labels, self.perform(task.generate_method, (task_name, self.corpus))], axis=1)

        features = CryptoUtils.compute_additionnal_features([
            lambda: self.features_extractor.add_feature_history_window_mlp(self.corpus, 2)
            # lambda: self.crypto_features_extractor.build_sequence_features(corpus, 1)
        ])

        CryptoUtils.resize_dataframes(self.corpus, features, labels)

        return features, labels
