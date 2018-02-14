import tensorflow as tf

from extractors.CryptoFeaturesExtractor import CryptoFeaturesExtractor as crypto_features_extractor
from controllers.CryptoFeaturesPreprocessor import CryptoFeaturesPreprocessor as crypto_features_preprocessor
from extractors.CryptoLabelsExtractor import CryptoLabelsExtractor as crypto_labels_extractor
from controllers.CryptoUtils import CryptoUtils


class CryptoDataConverter:
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
        crypto_features_preprocessor.preprocess_features(corpus)

        labels = CryptoUtils.compute_labels([
            lambda: crypto_labels_extractor.compute_variation_sign(corpus),
            lambda: crypto_labels_extractor.compute_next_price_at(corpus, 1),
            lambda: crypto_labels_extractor.compute_next_price_at(corpus, 2),
            lambda: crypto_labels_extractor.compute_next_price_at(corpus, 0)

        ])


        features = CryptoUtils.compute_additionnal_features([
            lambda: crypto_features_extractor.add_feature_history_window_mlp(corpus, 2)
            # lambda: self.crypto_features_extractor.build_sequence_features(corpus, 1)
        ])

        self.resize_dataframes(corpus, features, labels)

        return features, labels
