import pandas as pd


class FeaturesProcessor:

    def normalize_series(self, minbound, maxbound, series, centered=False):
        true_min = min_value = series.min()
        true_max = max_value = series.max()

        if centered:
            if -true_min[0] > max_value[0]:
                max_value = -true_min
            if -true_max[0] < min_value[0]:
                min_value = -true_max

        ratio = 0     # 0.000 - 1.000
        # ratio = 1/2 # 0.250 - 0.750
        # ratio =  1  # 0.333 - 0.666
        # ratio = 3/2 # 0.375 - 0.625
        min2 = min_value - ratio * (max_value - min_value)
        max2 = max_value + ratio * (max_value - min_value)

        return (maxbound-minbound) * ((series - min2) / (max2 - min2)) + minbound



    def create_context_window(self, window_size, features):
        context_features = pd.DataFrame()
        for column_name, column in features.items():
            for j in range(1, window_size + 1):
                context_features["w-" + str(j) + "_" + column_name] = column.shift(j)
        return context_features
