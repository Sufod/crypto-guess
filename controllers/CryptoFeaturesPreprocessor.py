class CryptoFeaturesPreprocessor:

    def preprocess_features(self, features):
        self.remove_unwanted_features(features)
        self.normalize_features(features)

    def remove_unwanted_features(self, features):
        # Remove time
        del (features['time'])

    def normalize_features(self, features):
        features['open'] = features['open'] * 0.01
        features['close'] = features['close'] * 0.01
        features['low'] = features['low'] * 0.01
        features['high'] = features['high'] * 0.01
        features['volumefrom'] = features['volumefrom'] * 0.0001
        features['volumeto'] = features['volumeto'] * 0.0001
