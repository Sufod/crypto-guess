import pandas as pd

from misc.Utils import Utils


class CryptoLabelsExtractor:

    def compute_variation_sign(self, args):

        corpus, task_name = self.get_corpus_and_task_name_from_args(args)

        variationList = []
        lastPrice = 0
        for i, row in enumerate(corpus.itertuples()):
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
        return Utils.transform_list_to_df(task_name, variationList)

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

    def compute_next_price_at(self, nb, args):
        corpus, task_name = self.get_corpus_and_task_name_from_args(args)
        future_prices = []
        for i, sample in enumerate(corpus.itertuples()):
            if i < nb:
                continue
            future_prices.append(sample.open)
        return Utils.transform_list_to_df(task_name, future_prices)

    def get_corpus_and_task_name_from_args(args):
        task_name = args[0]
        corpus = args[1]
        return corpus, task_name
