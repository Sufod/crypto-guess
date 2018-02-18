import os
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
        # Looking for first valid index in features and labels
        first_valid_index = Utils.get_first_valid_index(features, labels)
        # Cutting heads of labels and features to match first valid index
        features.drop(range(0, first_valid_index), inplace=True)
        labels.drop(range(0, first_valid_index), inplace=True)

        # Looking for last valid index in features and labels
        last_valid_index = Utils.get_last_valid_index(features, labels)
        # Cutting tails of all vectors to match features last valid index
        features.drop(features.tail(features.index[-1] - last_valid_index).index, inplace=True)
        labels.drop(labels.tail(labels.index[-1] - last_valid_index).index, inplace=True)

    @staticmethod
    def get_first_valid_index(features, labels):
        first_valid_index = -1
        for f_name, f in features.items():
            first_valid_index = max(first_valid_index, f.first_valid_index())
        for l_name, l in labels.items():
            first_valid_index = max(first_valid_index, l.first_valid_index())
        return first_valid_index

    @staticmethod
    def get_last_valid_index(features, labels):
        last_valid_index = float("inf")
        for f_name, f in features.items():
            last_valid_index = min(last_valid_index, f.last_valid_index())
        for l_name, l in labels.items():
            last_valid_index = min(last_valid_index, l.last_valid_index())
        return last_valid_index

    @staticmethod
    def get_dict_from_obj_list(obj_list):
        dict = {}
        for task in obj_list:
            dict[task.name] = task
        return dict

    @staticmethod
    def get_base_and_dir_names(filepath):
        basename = os.path.basename(filepath)
        dir_name = os.path.dirname(filepath)
        return basename, dir_name

    @staticmethod
    def get_sub_corpus_path(basename, dir_name, sub_corpus):
        sub_corpus_name = basename.replace("latest", sub_corpus)
        return dir_name + "/" + sub_corpus_name
