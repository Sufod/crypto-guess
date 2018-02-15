import pandas as pd

from controllers.CryptoDataConverter import CryptoDataConverter
from controllers.CryptoDataLoader import CryptoDataLoader

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator
from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops


class CryptoBrain:
    data_loader = CryptoDataLoader()

    def run(self, model, params):
        # Fetch the data
        corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-train.csv")
        data_converter = CryptoDataConverter(corpus)

        features, labels = data_converter.generate_features_and_labels(params)

        features = pd.concat([features, corpus], axis=1)

        # Feature columns describe how to use the input.
        model_feature_columns = []
        for feature in features.keys():
            model_feature_columns.append(tf.feature_column.numeric_column(key=feature))
        params['feature_columns'] = model_feature_columns

        classifier = tf.estimator.Estimator(model_fn=model.model_fn, params=params)

        # Train the model.
        params['learning_rate'] = 0.1
        params['batch_size'] = 100
        params['train_steps'] = 10000
        classifier.train(
            input_fn=lambda: data_converter.train_input_fn(features, labels, params['batch_size']),
            steps=params['train_steps'])
        # for task in range(params['n_tasks']):
        #    if params['n_classes'][task] == 1:
        predictions = classifier.predict(
            input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        self.show_prediction_graph(predictions, labels)

        # Re-Train the model.
        params['learning_rate'] = 0.001
        params['batch_size'] = 1
        params['train_steps'] = 10000
        classifier.train(
            input_fn=lambda: data_converter.train_input_fn(features, labels, params['batch_size']),
            steps=params['train_steps'])
        # for task in range(params['n_tasks']):
        #    if params['n_classes'][task] == 1:
        predictions = classifier.predict(
            input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        self.show_prediction_graph(predictions, labels)

        # Evaluate the model.
        corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-test.csv")
        data_converter = CryptoDataConverter(corpus)
        features, labels = data_converter.generate_features_and_labels(params)
        features = pd.concat([features, corpus], axis=1)
        eval_result = classifier.evaluate(
            input_fn=lambda: data_converter.eval_input_fn(features, labels, 1))

        if params['tasks']['l_price_at_0'].weight != 0:
            print('\nTest set  0 mse: {mse_l_price_at_0:0.3f}\n'.format(**eval_result))
        if params['tasks']['l_price_at_1'].weight != 0:
            print('\nTest set +1 mse: {mse_l_price_at_1:0.3f}\n'.format(**eval_result))
        if params['tasks']['l_price_at_2'].weight != 0:
            print('\nTest set +2 mse: {mse_l_price_at_2:0.3f}\n'.format(**eval_result))
        if params['tasks']['l_variation_sign'].weight != 0:
            print('\nTest set accuracy variation sign: {accuracy_l_variation_sign:0.3f}\n'.format(**eval_result))

        # Visualize predictions.

        # for task in range(params['n_tasks']):
        #    if params['n_classes'][task] == 1:
        predictions = classifier.predict(
            input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        self.show_prediction_graph(predictions, labels)

        # predictions = classifier.predict(
        #    input_fn=lambda: self.crypto_data_converter.eval_input_fn(features,labels=None,batch_size=1))
        # for pred_dict, expect_class, expect_regress1, expect_regress2 in zip(predictions, labels[0], labels[1], labels[2]):
        #    template = ('\nPrediction is "{}" / "{:.2f}" ({:.1f}%), expected "{}" / "{}"')
        #    class_id = pred_dict['class_ids_0'][0]
        #    probability = pred_dict['probabilities_0'][class_id]
        #    print(template.format(class_id, pred_dict['regressions_1'], pred_dict['regressions_2'],
        #                          100 * probability, expect_class, expect_regress1, expect_regress2))

    def show_prediction_graph(self, predictions, labels):
        lst_predict_1 = []
        lst_predict_2 = []
        lst_expect_1 = []
        lst_expect_2 = []
        for pred_dict, expect_regress_1, expect_regress_2 in zip(predictions, labels['l_price_at_1'],
                                                                 labels['l_price_at_0']):
            lst_predict_1.append(pred_dict['regressions_l_price_at_1'])
            lst_predict_2.append(pred_dict['regressions_l_price_at_2'])
            lst_expect_1.append(expect_regress_1)
            lst_expect_2.append(expect_regress_2)
        plt.plot(lst_predict_1, label="predict_l_price_at_1")
        plt.plot(lst_predict_2, label="predict_l_price_at_2")
        plt.plot(lst_expect_1, label="real+1")
        plt.plot(lst_expect_2, label="real")
        plt.legend(loc=0)
        plt.show()

    # def show_prediction_graph(self, predictions, labels):
    #     lst_predict_1 = []
    #     lst_predict_2 = []
    #     for pred_dict in predictions:
    #         lst_predict_1.append(pred_dict['regressions_l_price_at_1'])
    #         lst_predict_2.append(pred_dict['regressions_l_price_at_5'])
    #     plt.plot(lst_predict_1, label="predict_l_price_at_1")
    #     plt.plot(lst_predict_2, label="predict_l_price_at_5")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #     plt.show()
