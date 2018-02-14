import tensorflow as tf

from extractors.CryptoFeaturesExtractor import CryptoFeaturesExtractor
from controllers.CryptoFeaturesPreprocessor import CryptoFeaturesPreprocessor
from extractors.CryptoLabelsExtractor import CryptoLabelsExtractor
from controllers.CryptoUtils import CryptoUtils


class CryptoDataConverter:
    features_preprocessor = CryptoFeaturesPreprocessor()
    labels_extractor = CryptoLabelsExtractor()
    features_extractor = CryptoFeaturesExtractor()


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

    def generate_features_and_labels(self, corpus):

        self.features_preprocessor.preprocess_features(corpus)

        labels = CryptoUtils.compute_labels([
            lambda: self.labels_extractor.compute_variation_sign(corpus),
            lambda: self.labels_extractor.compute_next_price_at(corpus, 1),
            lambda: self.labels_extractor.compute_next_price_at(corpus, 2),
            lambda: self.labels_extractor.compute_next_price_at(corpus, 0)

        ])

        features = CryptoUtils.compute_additionnal_features([
            lambda: self.features_extractor.add_feature_history_window_mlp(corpus, 2)
            # lambda: self.crypto_features_extractor.build_sequence_features(corpus, 1)
        ])

        CryptoUtils.resize_dataframes(corpus, features, labels)

        return features, labels
