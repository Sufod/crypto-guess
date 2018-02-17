import pandas as pd


class FeaturesExtractor:

    # def add_feature_history_window_mlp(self, window_size, args):
    #     corpus=args[0]
    #     new_features = pd.DataFrame()
    #     for column_name, column in corpus.items():
    #         for j in range(1,window_size+1):
    #             new_features[column_name+"_at_-_"+str(j)] = column.shift(j)
    #     return new_features

    def compute_feature_at(self, target_feature_name, nb, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].shift(-nb)
        return new_column

    def compute_diff_feature(self, target1_feature_name, target2_feature_name, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[target1_feature_name] - features[target2_feature_name]
        return new_column

    def compute_variation_feature(self, target_feature_name, nb, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        current = features[target_feature_name]
        previous = features[target_feature_name].shift(-nb)
        new_column[feature_name] = current - previous
        return new_column

    # def build_sequence_features(self, features, window_size):
    #     sequence_features = pd.DataFrame()
    #     for column_name, column in features.items():
    #         sequence_features[column_name]=
    #         for j in range(0, window_size+1):
    #             sequence_features[column_name].append(column.shift(j))
    #     return sequence_features

    # def add_feature_history_window(self, dataframe, window_size):
    #     samples_history = []
    #     samples_history_list = []
    #     for i, sample in enumerate(dataframe.itertuples()):
    #         if i < window_size:
    #             samples_history.append(sample)
    #             continue
    #         samples_history_list.append(list(samples_history))
    #         samples_history.pop(0)
    #         samples_history.append(sample)
    #     dataframe.drop(dataframe.tail(window_size).index, inplace=True)
    #     dataframe.reset_index(drop=True,inplace=True)
    #     return self.add_feature_to_dataframe(dataframe, 'SampleHistory', samples_history_list)

    def add_feature_to_dataframe(self, dataframe, feature_name, feature_list):
        return pd.concat([dataframe, pd.DataFrame(data={feature_name: pd.Series(data=feature_list)})], axis=1)
