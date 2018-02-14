import pandas as pd

from controllers.CryptoUtils import CryptoUtils

class CryptoLabelsExtractor:

    def compute_labels(self,method_list):
        return CryptoUtils.compute_methods(method_list,cut_mode=True)

    def reduce_labels(self, labels, mini):
        if len(labels) - mini > 0:
            labels = self.remove_rows_from_labels(labels, len(labels) - mini)
            labels.reset_index(drop=True, inplace=True)
        return labels

    def compute_variation_sign(self, features):
        variationList = []
        lastPrice = 0
        for i, row in enumerate(features.itertuples()):
            if i == 0:
                lastPrice = row.open
                continue
            diff = row.open - lastPrice
            if diff > 0.0:
                variationList.append(1)
            # elif diff == 0.0:
            #     variationList.append(1)
            else:
                variationList.append(0)
            lastPrice = row.open
        return CryptoUtils.transform_list_to_df('l_variation_sign',variationList)

    def compute_current_price(self, features):
        openList = []
        rows_to_remove = 0
        for i, row in enumerate(features.itertuples()):
            openList.append(row.open)
        labels = pd.Series(data=openList)
        return labels, rows_to_remove

    def compute_next_price(self, features):
        openList = []
        for i, row in enumerate(features.itertuples()):
            if i == 0: continue
            openList.append(row.open)
        labels = pd.Series(data=openList)
        return labels

    def compute_next_price_at(self, features, nb):
        future_prices = []
        for i, sample in enumerate(features.itertuples()):
            if i < nb:
                continue
            future_prices.append(sample.open)
        return CryptoUtils.transform_list_to_df('l_price_at_' + str(nb),future_prices)

    def remove_rows_from_labels(self, labels, window_size):
        if isinstance(labels, tuple):
            newLabels = []
            for i, label in enumerate(labels):
                newLabels.append(label[:-window_size or None].reset_index(drop=True))
            labels = tuple(newLabels)
        else:
            labels = labels[:-window_size].reset_index(drop=True)
        return labels
