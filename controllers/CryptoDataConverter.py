import tensorflow as tf

from controllers.CryptoFeaturesExtractor import CryptoFeaturesExtractor
from controllers.CryptoFeaturesPreprocessor import CryptoFeaturesPreprocessor
from controllers.CryptoLabelsExtractor import CryptoLabelsExtractor


class CryptoDataConverter:
    crypto_features_extractor = CryptoFeaturesExtractor()
    crypto_labels_extractor = CryptoLabelsExtractor()
    crypto_features_preprocessor = CryptoFeaturesPreprocessor()

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
        self.crypto_features_preprocessor.preprocess_features(corpus)
        labels = self.crypto_labels_extractor.compute_labels([
            lambda: self.crypto_labels_extractor.compute_variation_sign(corpus),
            lambda: self.crypto_labels_extractor.compute_next_price_at(corpus, 1),
            lambda: self.crypto_labels_extractor.compute_next_price_at(corpus, 5),
            lambda: self.crypto_labels_extractor.compute_next_price_at(corpus, 0)

        ])
        features = self.crypto_features_extractor.compute_additionnal_features([
            lambda: self.crypto_features_extractor.add_feature_history_window_mlp(corpus, 1)
        ])

        self.resize_dataframes(corpus, features, labels)

        return features, labels

    def resize_dataframes(self, corpus, features, labels):
        labels_size = labels.shape[0]
        corpus.drop(range(labels_size, corpus.shape[0]), inplace=True)
        features.drop(range(labels_size, features.shape[0]), inplace=True)
        top_index = -1
        for column_name, column in features.items():
            top_index = max(top_index, column.first_valid_index())
        if top_index > 0:
            corpus.drop(range(0, top_index), inplace=True)
            features.drop(range(0, top_index), inplace=True)
            labels.drop(range(0, top_index), inplace=True)
