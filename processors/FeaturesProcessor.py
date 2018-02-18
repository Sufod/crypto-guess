class FeaturesProcessor:

    def preprocess_features(self, features):
        self.normalize_features(features)

    def normalize_features(self, features):
        for feature_name in features.keys():
            if feature_name != 'l_real_price':
                min = features[feature_name].min()
                max = features[feature_name].max()
                ratio = 0 # 0 - 1
                # ratio = 1/2  # 0.25 - 0.75
                # ratio = 1  # 0 - 1
                # ratio = 3/2 #0.33 - 0.66
                min2 = min - ratio * (max - min)
                max2 = max + ratio * (max - min)
                features[feature_name] = ((features[feature_name] - min2) / (max2 - min2))

        # features['open'] = features['open'] * 0.01
        # features['close'] = features['close'] * 0.01
        # features['low'] = features['low'] * 0.01
        # features['high'] = features['high'] * 0.01
        # features['volumefrom'] = features['volumefrom'] * 0.0001
        # features['volumeto'] = features['volumeto'] * 0.0001
