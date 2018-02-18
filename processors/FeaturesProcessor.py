class FeaturesProcessor:

    @staticmethod
    def normalize_series(series):
        min_value = series.min()
        max_value = series.max()
        ratio = 0  # 0 - 1
        # ratio = 1/2  # 0.25 - 0.75
        # ratio = 1  # 0 - 1
        # ratio = 3/2 #0.33 - 0.66
        min2 = min_value - ratio * (max_value - min_value)
        max2 = max_value + ratio * (max_value - min_value)
        return (series - min2) / (max2 - min2)

        # features['open'] = features['open'] * 0.01
        # features['close'] = features['close'] * 0.01
        # features['low'] = features['low'] * 0.01
        # features['high'] = features['high'] * 0.01
        # features['volumefrom'] = features['volumefrom'] * 0.0001
        # features['volumeto'] = features['volumeto'] * 0.0001
