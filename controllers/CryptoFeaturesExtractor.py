import pandas as pd

from controllers.CryptoUtils import CryptoUtils

class CryptoFeaturesExtractor:

    def compute_additionnal_features(self, method_list):
        return CryptoUtils.compute_methods(method_list,cut_mode=False)

    def add_feature_history_window_mlp(self, features, window_size):
        new_features = pd.DataFrame()
        for column_name, column in features.items():
            for j in range(1,window_size+1):
                new_features[column_name+"_at_-_"+str(j)] = column.shift(j)
        return new_features

    # def build_sequence_features(self, features, window_size):
    #     sequence_features = pd.DataFrame()
    #     for column_name, column in features.items():
    #         sequence_features[column_name]=
    #         for j in range(0, window_size+1):
    #             sequence_features[column_name].append(column.shift(j))
    #     return sequence_features

    def remove_rows_from_labels(self, labels, window_size):
        if isinstance(labels, tuple):
            newLabels = []
            for i, label in enumerate(labels):
                newLabels.append(label[window_size:].reset_index(drop=True))
            labels = tuple(newLabels)
        else:
            labels = labels[window_size:].reset_index(drop=True)
        return labels

    def add_feature_history_window(self, dataframe, window_size):
        samples_history = []
        samples_history_list = []
        for i, sample in enumerate(dataframe.itertuples()):
            if i < window_size:
                samples_history.append(sample)
                continue
            samples_history_list.append(list(samples_history))
            samples_history.pop(0)
            samples_history.append(sample)
        dataframe.drop(dataframe.tail(window_size).index, inplace=True)
        dataframe.reset_index(drop=True,inplace=True)
        return self.add_feature_to_dataframe(dataframe, 'SampleHistory', samples_history_list)

    def add_feature_to_dataframe(self, dataframe, feature_name, feature_list):
        return pd.concat([dataframe, pd.DataFrame(data={feature_name: pd.Series(data=feature_list)})], axis=1)