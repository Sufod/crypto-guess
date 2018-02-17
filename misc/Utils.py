import pandas as pd


class Utils:

    @staticmethod
    def compute_labels(method_list):
        return Utils.compute_methods(method_list, cut_mode=True)

    @staticmethod
    def compute_additionnal_features(method_list):
        return Utils.compute_methods(method_list, cut_mode=False)

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
    def resize_dataframes(features, labels):
        # Cutting tails of features to match labels_size
        labels_size = labels.shape[0]
        features.drop(range(labels_size, features.shape[0]), inplace=True)

        # Cutting heads of all vectors to match features first valid index
        first_valid_index = -1
        for column_name, column in features.items():
            first_valid_index = max(first_valid_index, column.first_valid_index())
        features.drop(range(0, first_valid_index), inplace=True)
        labels.drop(range(0, first_valid_index), inplace=True)

        # Cutting tails of all vectors to match features last valid index
        last_valid_index = float("inf")
        for column_name, column in features.items():
            last_valid_index = min(last_valid_index, column.last_valid_index())
        features.drop((range(last_valid_index, column.shape[0])), inplace=True)
        labels.drop(range(last_valid_index, column.shape[0]), inplace=True)

    @staticmethod
    def get_dict_from_obj_list(obj_list):
        dict = {}
        for task in obj_list:
            dict[task.name] = task
        return dict
