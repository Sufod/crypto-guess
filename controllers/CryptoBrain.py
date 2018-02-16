import pandas as pd

from controllers.CryptoDataConverter import CryptoDataConverter
from controllers.CryptoDataLoader import CryptoDataLoader

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator

from tasks.RegressionTask import RegressionTask


class CryptoBrain:
    data_loader = CryptoDataLoader()

    def run(self, model, params):

        # Fetch the training data
        train_corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-train.csv")
        train_data_converter = CryptoDataConverter(train_corpus)
        train_features, train_labels = train_data_converter.generate_features_and_labels(params)

        # Fetch the validation data
        dev_corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-dev.csv")
        dev_data_converter = CryptoDataConverter(dev_corpus)
        dev_features, dev_labels = dev_data_converter.generate_features_and_labels(params)

        # Fetch the test data
        test_corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-test.csv")
        test_data_converter = CryptoDataConverter(test_corpus)
        test_features, test_labels = test_data_converter.generate_features_and_labels(params)

        Z = 0
        for task_name in params['tasks'].keys():
            Z += params['tasks'][task_name].weight
        for task_name in params['tasks'].keys():
            params['tasks'][task_name].weight /= Z

        # Feature columns describe how to use the input.
        model_feature_columns = []
        for feature in train_features.keys():
            model_feature_columns.append(tf.feature_column.numeric_column(key=feature))
        params['feature_columns'] = model_feature_columns

        classifier = tf.estimator.Estimator(model_fn=model.model_fn, params=params)

        #
        #
        # Training
        visualize = False
        for i in range(params['supervision_steps']):
            #
            # Train the model on train set.
            classifier.train(
                input_fn=lambda: train_data_converter.train_input_fn(train_features, train_labels, params['batch_size']),
                steps=params['train_steps'] / params['supervision_steps'])
            #
            # Evaluate the model on train set.
            self.evaluate(classifier, params, train_features, train_labels, train_data_converter, False, 'Train')
            #
            # Evaluate the model on dev set.
            self.evaluate(classifier, params, dev_features, dev_labels, dev_data_converter, visualize, 'Dev')

        #
        #
        # Evaluate the model on test set.
        self.evaluate(classifier, params, test_features, test_labels, test_data_converter, True, 'Test')

    def evaluate(self, classifier, params, features, labels, data_converter, visualize=False, mode='Train'):
        eval_result = classifier.evaluate(
            input_fn=lambda: data_converter.eval_input_fn(features, labels, 1))
        for task_name in params['tasks'].keys():
            if params['tasks'][task_name].weight != 0:
                if task_name == 'l_price_at_0':
                    print('\n' + mode + ' set  0 mse: {mse_l_price_at_0:0.8f}\n'.format(**eval_result))
                if task_name == 'l_price_at_1':
                    print('\n' + mode + ' set +1 mse: {mse_l_price_at_1:0.8f}\n'.format(**eval_result))
                if task_name == 'l_price_at_2':
                    print('\n' + mode + ' set +2 mse: {mse_l_price_at_2:0.8f}\n'.format(**eval_result))
                if task_name == 'l_variation_sign':
                    print('\n' + mode + ' set accuracy variation sign: {accuracy_l_variation_sign:0.3f}\n'.format(
                        **eval_result))

        if visualize:
            predictions = classifier.predict(
                input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
            self.show_prediction_graph(predictions, labels, params)


    def show_prediction_graph(self, predictions, labels, params):

        lst_predict = {}
        lst_real = []
        lst_expect = []

        for task_name in params['tasks'].keys():
            if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                lst_predict[task_name] = []

        for pred_dict, expect, real in zip(predictions, labels['l_price_at_1'], labels['l_price_at_0']):
            for task_name in params['tasks'].keys():
                if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                    lst_predict[task_name].append(pred_dict['regressions_' + task_name])
            lst_expect.append(expect)
            lst_real.append(real)

        for task_name in params['tasks'].keys():
            if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                plt.plot(lst_predict[task_name], label="predict_" + task_name)

        plt.plot(lst_expect, label="real+1")
        plt.plot(lst_real, label="real")
        plt.legend(loc=0)
        plt.show()
