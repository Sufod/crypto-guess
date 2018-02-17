import argparse
import os
import numpy as np

from loaders.DataLoader import DataLoader
from misc.Logger import Logger


class CorpusUtils:

    @staticmethod
    def produce_train_dev_test_from_full_corpus(filepath):
        basename = os.path.basename(filepath)
        train_name = basename.replace("latest", "train")
        dev_name = basename.replace("latest", "dev")
        test_name = basename.replace("latest", "test")
        corpus = DataLoader.load_csv_data(filepath)
        train, dev, test = np.split(corpus, [int(.8 * len(corpus)), int(.9 * len(corpus))])
        dirname = os.path.dirname(filepath)
        filepath_train_csv = dirname + "/" + train_name
        filepath_dev_csv = dirname + "/" + dev_name
        filepath_test_csv = dirname + "/" + test_name
        print(filepath_train_csv)
        train.to_csv(path_or_buf=filepath_train_csv, index=False)
        dev.to_csv(path_or_buf=filepath_dev_csv, index=False)
        test.to_csv(path_or_buf=filepath_test_csv, index=False)


def main():
    DEFAULT_ACTION = "split"
    # ---- Parsing args ----#
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath",
                        metavar="F",
                        help="Specify the path of the file containing csv corpus"
                        )
    parser.add_argument("-a", "--action",
                        help="choose the action to perform on csv corpus (default:" + DEFAULT_ACTION + ") Possible actions are 'split'",
                        default=os.environ.get('DEFAULT_ACTION', DEFAULT_ACTION),
                        )

    args = parser.parse_args()

    action = args.action
    filepath = args.filepath
    if action == 'split':
        CorpusUtils.produce_train_dev_test_from_full_corpus(filepath)
    exit(0)


if __name__ == "__main__":
    main()
