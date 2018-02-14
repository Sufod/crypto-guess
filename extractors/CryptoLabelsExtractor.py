import pandas as pd

from controllers.CryptoUtils import CryptoUtils

class CryptoLabelsExtractor:

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
        for i, row in enumerate(features.itertuples()):
            openList.append(row.open)
        labels = pd.Series(data=openList)
        return labels

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
