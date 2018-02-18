import tensorflow as tf
import pandas as pd

from processors.FeaturesProcessor import FeaturesProcessor
from processors.FeaturesExtractor import FeaturesExtractor

from misc.Utils import Utils
from features.CorpusFeature import CorpusFeature
from tasks.RegressionTask import RegressionTask


class DataConverter:
    features_preprocessor = FeaturesProcessor()
    features_extractor = FeaturesExtractor()

    def __init__(self, corpus):
        self.corpus = corpus

    def perform(self, fun, *args):
        return fun(*args)

    def generate_features_and_labels(self, params):
        # self.features_preprocessor.preprocess_features(self.corpus)
        features_params = self.extract_features_params(params["features"])
        tasks_params = self.extract_tasks_params(params["tasks"])

        labels = self.extract_labels(tasks_params)
        features = self.extract_features(features_params)

        Utils.resize_dataframes(features, labels)
        self.normalize_labels(labels, tasks_params)
        self.normalize_features(features, features_params)
        return features, labels

    def normalize_labels(self, labels, tasks_params):
        tasks_dict = tasks_params[1]
        for task_name, task in tasks_dict.items():
            if isinstance(task, RegressionTask) and task.normalization is not False:
                labels[task_name] = task.normalization(labels[task_name])

    def normalize_features(self, features, features_params):
        features_dict = features_params[2]
        for feature_name, feature in features_dict.items():
            if feature.normalization is not False:
                features[feature_name] = feature.normalization(features[feature_name])

    def extract_features(self, features_params):
        corpus_features_names = features_params[0]
        features = self.get_corpus_features(corpus_features_names)
        features_names = features_params[1]
        features_dict = features_params[2]
        for feature_name in features_names:
            feature_param = features_dict[feature_name]
            feature = self.perform(feature_param.generate_method, (feature_name, features))
            if feature_param.inflow_normalization is not False:
                feature = feature_param.inflow_normalization(feature)
            features = pd.concat([features, feature], axis=1)
        return features

    def get_corpus_features(self, corpus_features_names):
        for corpus_column in self.corpus.keys():
            if corpus_column not in corpus_features_names:
                del (self.corpus[corpus_column])
        return self.corpus

    def extract_labels(self, tasks_params):
        tasks_names = tasks_params[0]
        tasks_dict = tasks_params[1]
        labels = pd.DataFrame()
        tmp_labels = self.corpus
        for task_name in tasks_names:
            task = tasks_dict[task_name]
            label = self.perform(task.generate_method, (task_name, tmp_labels))
            if isinstance(task, RegressionTask) and task.inflow_normalization is not False:
                label = task.inflow_normalization(label)
            tmp_labels = pd.concat([tmp_labels, label], axis=1)
            labels = pd.concat([labels, label], axis=1)
        return labels

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

    @staticmethod
    def extract_features_params(features_list):
        corpus_features_names = []
        features_names = []
        features_dict = {}
        for feature in features_list:
            features_dict[feature.name] = feature
            if isinstance(feature, CorpusFeature):
                corpus_features_names.append(feature.name)
            else:
                features_names.append(feature.name)
        return corpus_features_names, features_names, features_dict

    @staticmethod
    def extract_tasks_params(tasks_list):
        tasks_names = []
        tasks_dict = {}
        for task in tasks_list:
            tasks_dict[task.name] = task
            tasks_names.append(task.name)
        return tasks_names, tasks_dict
