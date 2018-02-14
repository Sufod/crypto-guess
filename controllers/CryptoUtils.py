import pandas as pd

class CryptoUtils:

    @staticmethod
    def compute_labels(method_list):
        return CryptoUtils.compute_methods(method_list,cut_mode=True)

    @staticmethod
    def compute_additionnal_features(method_list):
        return CryptoUtils.compute_methods(method_list,cut_mode=False)

    @staticmethod
    def transform_list_to_df(name, list):
        return pd.DataFrame(data=list, columns=[name])

    @staticmethod
    def compute_methods(method_list, cut_mode=False):
        df_list = []
        min_size = float("inf")
        for method in method_list:
            new_df = method()
            min_size = min(new_df.shape[0], min_size)
            df_list.append(new_df)
        new_df = pd.DataFrame()
        for df in df_list:
            if cut_mode:
                df.drop(df.tail(df.shape[0] - min_size).index, inplace=True)
            new_df = pd.concat([new_df, df], axis=1)
        return new_df

    @staticmethod
    def resize_dataframes(corpus, features, labels):
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
