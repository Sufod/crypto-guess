import tensorflow as tf
import pandas as pd

from processors.FeaturesProcessor import FeaturesProcessor
from processors.FeaturesExtractor import FeaturesExtractor

from misc.Utils import Utils
from features.CorpusFeature import CorpusFeature


class DataConverter:
    features_preprocessor = FeaturesProcessor()
    features_extractor = FeaturesExtractor()

    @staticmethod
    def train_input_fn(features, labels, batch_size):
        """An input function for training"""

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), dict(labels)))

        # Shuffle, repeat, and batch the examples.
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    @staticmethod
    def eval_input_fn(features, labels, batch_size):
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

    def __init__(self, corpus):
        self.corpus = corpus

    def perform(self, fun, *args):
        return fun(*args)

    def generate_features_and_labels(self, params):
        # self.features_preprocessor.preprocess_features(self.corpus)
        features_params = self.extract_features_params(params)
        labels = self.extract_labels(params)
        features = self.extract_features(features_params)
        Utils.resize_dataframes(features, labels)
        self.features_preprocessor.preprocess_features(features)
        self.features_preprocessor.preprocess_features(labels)

        return features, labels

    def extract_features(self, features_params):
        features = self.get_corpus_features(features_params[0])
        for feature_name in features_params[1]:
            feature = features_params[2][feature_name]
            features = pd.concat([
                features,
                self.perform(feature.generate_method, (feature_name, features))
            ], axis=1)
        return features

    def get_corpus_features(self, corpus_features_names):
        for corpus_column in self.corpus.keys():
            if corpus_column not in corpus_features_names:
                del (self.corpus[corpus_column])
        return self.corpus

    def extract_labels(self, params):
        if isinstance(params["tasks"],list):
            params["tasks"] = Utils.get_dict_from_obj_list(params["tasks"])
        df_labels = pd.DataFrame()
        labels = []
        for task_name, task in params["tasks"].items():
            labels.append(self.perform(task.generate_method, (task_name, self.corpus)))
        # Cutting tail of all vectors until labels last valid index
        last_index = float("inf")
        for column in labels:
            last_index = min(last_index, column.last_valid_index())
        for column in labels:
            test = column.drop(range(last_index, column.shape[0]))
            df_labels = pd.concat([df_labels, test], axis=1)
        return df_labels

    def extract_features_params(self, params):
        corpus_features_names = []
        features_names = []
        features_params = {}
        for feature in params["features"]:
            features_params[feature.name] = feature
            if isinstance(feature, CorpusFeature):
                corpus_features_names.append(feature.name)
            else:
                features_names.append(feature.name)
        return corpus_features_names, features_names, features_params
