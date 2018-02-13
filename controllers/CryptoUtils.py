import pandas as pd

class CryptoUtils:

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
