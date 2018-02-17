import pandas as pd
import numpy as np


class FeaturesExtractor:
    
    def add_feature_to_dataframe(self, dataframe, feature_name, feature_list):
        return pd.concat([dataframe, pd.DataFrame(data={feature_name: pd.Series(data=feature_list)})], axis=1)
    
    #
    #
    # data retrieval
    def compute_feature_at(self, target_feature_name, nb, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].shift(-nb)
        return new_column
    
    #
    #
    # arithmetic transformation between features
    def compute_arithmetic_feature(self, f1_name, op, f2_name, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()

        if op == 'add':
            new_column[feature_name] = self.add(f1_name, f2_name, args)
        elif op == 'sub':
            new_column[feature_name] = self.sub(f1_name, f2_name, args)
        elif op == 'mul':
            new_column[feature_name] = self.mul(f1_name, f2_name, args)
        elif op == 'div':
            new_column[feature_name] = self.div(f1_name, f2_name, args)
        elif op == 'cadd':
            new_column[feature_name] = self.add_cst(f1_name, f2_name, args)
        elif op == 'cmul':
            new_column[feature_name] = self.mul_cst(f1_name, f2_name, args)
        else:
            print("\nError on features arithmetic computations\n"
                  "Allowed operators : add, sub, mul, div, cadd, cmul\n"
                  "1st feature used as substitution\n")
            new_column = features[f1_name]

        return new_column

    def opp(self, target_feature_name, args):
        feature_name = args[0]
        new_column=pd.DataFrame()
        new_column[feature_name] = self.cst_sub(target_feature_name, 0, args)
        return new_column

    def inv(self, target_feature_name, args):
        feature_name = args[0]
        new_column=pd.DataFrame()
        new_column[feature_name] = self.cst_div(target_feature_name, 1, args)
        return new_column
    
    def add(self, f1_name, f2_name, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[f1_name].add(features[f2_name], axis=0, level=None, fill_value=None)
        return new_column
    
    def sub(self, f1_name, f2_name, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[f1_name].sub(features[f2_name], axis=0, level=None, fill_value=None)
        return new_column
    
    def mul(self, f1_name, f2_name, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[f1_name].mul(features[f2_name], axis=0, level=None, fill_value=None)
        return new_column
    
    def div(self, f1_name, f2_name, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[f1_name].div(features[f2_name], axis=0, level=None, fill_value=None)

        for index, item in new_column[feature_name].items():
            if np.isnan(item) and not np.isnan(features[f1_name][index]) and not np.isnan(features[f2_name][index]):
                new_column[feature_name][index] = 1

        min = new_column[feature_name].min()
        max = new_column[feature_name].max()

        if min == float('-inf') or max == float('inf'):
            sorted_feature = new_column[feature_name].sort_values()
            for index, item in sorted_feature.items():
                if item != float('-inf'):
                    min = item
                    break
            for index, item in sorted_feature.items():
                if item == float('-inf'):
                    new_column[feature_name][index] = min
                else:
                    break

            sorted_feature = sorted_feature[::-1]
            for index, item in sorted_feature.items():
                if item != float('inf'):
                    max = item
                    break
            for index, item in sorted_feature.items():
                if item == float('inf'):
                    new_column[feature_name][index] = max
                else:
                    break

        return new_column

    def add_cst(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].add(cst, axis=0, level=None, fill_value=None)
        return new_column

    def sub_cst(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].sub(cst, axis=0, level=None, fill_value=None)
        return new_column

    def cst_sub(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].rsub(cst, axis=0, level=None, fill_value=None)
        return new_column
    
    def mul_cst(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].mul(cst, axis=0, level=None, fill_value=None)
        return new_column

    def div_cst(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column = pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].div(cst, axis=0, level=None, fill_value=None)
        return new_column

    def cst_div(self, target_feature_name, cst, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        new_column[feature_name] = features[target_feature_name].rdiv(cst, axis=0, level=None, fill_value=None)

        for index, item in new_column[feature_name].items():
            if np.isnan(item) and not np.isnan(features[target_feature_name][index]):
                new_column[feature_name][index] = 1

        min = new_column[feature_name].min()
        max = new_column[feature_name].max()

        if min == float('-inf') or max == float('inf'):
            sorted_feature = new_column[feature_name].sort_values()
            for index, item in sorted_feature.items():
                if item != float('-inf'):
                    min = item
                    break
            for index, item in sorted_feature.items():
                if item == float('-inf'):
                    new_column[feature_name][index] = min
                else:
                    break

            sorted_feature = sorted_feature[::-1]
            for index, item in sorted_feature.items():
                if item != float('inf'):
                    max = item
                    break
            for index, item in sorted_feature.items():
                if item == float('inf'):
                    new_column[feature_name][index] = max
                else:
                    break

        return new_column


    #
    #
    # analytic transformation of one feature
    def compute_variation_feature(self, target_feature_name, nb, args):
        feature_name = args[0]
        features = args[1]
        new_column=pd.DataFrame()
        current = features[target_feature_name]
        previous = features[target_feature_name].shift(-nb)
        new_column[feature_name] = current - previous
        return new_column

    def mean(self, target_feature_name, nb, args):

        feature_name = args[0]
        features = args[1]

        window = pd.DataFrame()
        window[feature_name + '_0'] = features[target_feature_name]
        for i in range(1, -nb):
            window[feature_name + '_' + str(i)] = features[target_feature_name].shift(i)

        new_column = pd.DataFrame()
        new_column[feature_name] = window.mean(1)
        for i in range(0, -(nb+1)):
            new_column[feature_name][i] = float('nan')

        return new_column

    def min(self, target_feature_name, nb, args):

        feature_name = args[0]
        features = args[1]

        window = pd.DataFrame()
        window[feature_name + '_0'] = features[target_feature_name]
        for i in range(1, -nb):
            window[feature_name + '_' + str(i)] = features[target_feature_name].shift(i)

        new_column = pd.DataFrame()
        new_column[feature_name] = window.min(1)
        for i in range(0, -(nb+1)):
            new_column[feature_name][i] = float('nan')

        return new_column

    def max(self, target_feature_name, nb, args):

        feature_name = args[0]
        features = args[1]

        window = pd.DataFrame()
        window[feature_name + '_0'] = features[target_feature_name]
        for i in range(1, -nb):
            window[feature_name + '_' + str(i)] = features[target_feature_name].shift(i)

        new_column = pd.DataFrame()
        new_column[feature_name] = window.max(1)
        for i in range(0, -(nb+1)):
            new_column[feature_name][i] = float('nan')

        return new_column


    # def add_feature_history_window_mlp(self, window_size, args):
    #     corpus=args[0]
    #     new_features = pd.DataFrame()
    #     for column_name, column in corpus.items():
    #         for j in range(1,window_size+1):
    #             new_features[column_name+"_at_-_"+str(j)] = column.shift(j)
    #     return new_features

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
