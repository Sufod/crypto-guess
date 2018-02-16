class FeaturesProcessor:

    def preprocess_features(self, features):
        self.normalize_features(features)

    def normalize_features(self, features):
        for feature_name in features.keys():
            min = features[feature_name].min()
            max = features[feature_name].max()
            min2 = min - 3 * (max - min) / 2
            max2 = max + 3 * (max - min) / 2
            features[feature_name] = ( (features[feature_name] - min2) / (max2 - min2) )

        # features['open'] = features['open'] * 0.01
        # features['close'] = features['close'] * 0.01
        # features['low'] = features['low'] * 0.01
        # features['high'] = features['high'] * 0.01
        # features['volumefrom'] = features['volumefrom'] * 0.0001
        # features['volumeto'] = features['volumeto'] * 0.0001
