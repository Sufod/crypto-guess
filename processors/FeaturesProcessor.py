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
