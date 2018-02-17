from evaluate.CryptoGambler import CryptoGambler
from loaders.DataLoader import DataLoader

import tensorflow as tf
import matplotlib.pyplot as plt

from misc.Logger import Logger
from processors.DataConverter import DataConverter
from tasks.RegressionTask import RegressionTask


class CryptoBrain:

    def run(self, model, params):
        data_loader = DataLoader()
        # Fetch the training data
        train_features, train_labels = data_loader.load_train_dataframes(params)

        # Fetch the validation data
        dev_features, dev_labels = data_loader.load_dev_dataframes(params)

        # Fetch the test data
        test_features, test_labels = data_loader.load_test_dataframes(params)

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
                input_fn=lambda: DataConverter.train_input_fn(train_features, train_labels, params['batch_size']),
                steps=params['train_steps'] / params['supervision_steps'])
            #
            # Evaluate the model on train set.
            self.evaluate(classifier, params, train_features, train_labels, False, 'Train')
            #
            # Evaluate the model on dev set.
            self.evaluate(classifier, params, dev_features, dev_labels, visualize, 'Dev')

        #
        #
        # Evaluate the model on test set.
        self.evaluate(classifier, params, test_features, test_labels, True, 'Test')

        # Gamble

        predictions = classifier.predict(
            input_fn=lambda: DataConverter.eval_input_fn(test_features, labels=None, batch_size=1))

        l_real_price = test_labels['l_real_price']
        first_valid_index = l_real_price[l_real_price.first_valid_index()]
        crypto_gambler = CryptoGambler(first_valid_index)
        crypto_gambler_oracle = CryptoGambler(first_valid_index)
        crypto_gambler_bad = CryptoGambler(first_valid_index)
        crypto_gambler_baseline = CryptoGambler(first_valid_index)
        last_predict = 0.0
        last_real = 0.0
        for pred_dict, real in zip(predictions, l_real_price):
            if last_predict == 0.0:
                last_real = real
                last_predict = pred_dict['regressions_l_price_at_1']
            else:
                oracle_predicted_diff = real - last_real
                predicted_diff = pred_dict['regressions_l_price_at_1'] - last_predict

                crypto_gambler_oracle.evaluate_new_sample(last_real, oracle_predicted_diff)
                crypto_gambler.evaluate_new_sample(real, predicted_diff)
                crypto_gambler_bad.evaluate_new_sample(last_real, -oracle_predicted_diff)
                crypto_gambler_baseline.evaluate_new_sample(real, oracle_predicted_diff)

                last_predict = pred_dict['regressions_l_price_at_1']
                last_real = real

        labels_shape = l_real_price[test_labels.shape[0]]

        Logger.bold("Bad Gambler:")
        crypto_gambler_bad.get_evaluation_results(labels_shape)
        Logger.bold("Baseline Gambler:")
        crypto_gambler_baseline.get_evaluation_results(labels_shape)
        Logger.bold("Eval Gambler:")
        crypto_gambler.get_evaluation_results(labels_shape)
        Logger.bold("Oracle Gambler:")
        crypto_gambler_oracle.get_evaluation_results(labels_shape)

    def evaluate(self, classifier, params, features, labels, visualize=False, mode='Train'):
        eval_result = classifier.evaluate(
            input_fn=lambda: DataConverter.eval_input_fn(features, labels, 1))
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
                input_fn=lambda: DataConverter.eval_input_fn(features, labels=None, batch_size=1))
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
