from evaluate.CryptoGambler import CryptoGambler
from features.SymbolicFeature import SymbolicFeature
from loaders.DataLoader import DataLoader

import tensorflow as tf
import matplotlib.pyplot as plt

from misc.Logger import Logger
from misc.Utils import Utils
from processors.DataConverter import DataConverter
from tasks.RegressionTask import RegressionTask

L_VALUE = 'l_open'
L_MEAN = 'l_open_mean'
L_VAR = 'l_open_var'
L_VARSIGN = 'l_open_varsign'
TIME_FRAME = 3

class CryptoBrain:

    def run(self, model, params):

        with DataLoader(params) as data_loader:
            # Fetch the training data
            train_features, train_labels = data_loader.load_train_dataframes()

            # Fetch the validation data
            dev_features, dev_labels = data_loader.load_dev_dataframes()

            # Fetch the test data
            test_features, test_labels = data_loader.load_test_dataframes()

        params["tasks"] = Utils.get_dict_from_obj_list(params["tasks"])
        Z = 0
        for task_name in params['tasks'].keys():
            Z += params['tasks'][task_name].weight
        for task_name in params['tasks'].keys():
            params['tasks'][task_name].weight /= Z

        # Feature columns describe how to use the input.
        model_feature_columns = []
        for feature in params['features']:
            if isinstance(feature, SymbolicFeature):
                x = tf.feature_column.categorical_column_with_vocabulary_list(key=feature.name,
                                                                              vocabulary_list=feature.vocabulary,
                                                                              num_oov_buckets=2)
                y = tf.feature_column.embedding_column(x, dimension=feature.embedding_units)
                model_feature_columns.append(y)
            else:
                model_feature_columns.append(tf.feature_column.numeric_column(key=feature.name))
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
            # Gamble on dev set.
            self.gambling(classifier, dev_features, dev_labels)
            # params['tasks']['l_price_at_0'].weight *= (i+1)/(i+2)
            # params['tasks'][L_PRICE].weight *= (i+2)/(i+1)
            # params['tasks'][L_VARIATION].weight *= (i+2)/(i+1)


        #
        #
        # Evaluate the model on test set.
        self.evaluate(classifier, params, test_features, test_labels, True, 'Test')

        #
        #
        # Gamble
        self.gambling(classifier, test_features, test_labels)


    def gambling(self, classifier, test_features, test_labels):
        predictions = classifier.predict(
            input_fn=lambda: DataConverter.eval_input_fn(test_features, labels=None, batch_size=1))

        l_real_value = test_labels['l_current_open']
        first_valid_index = l_real_value[l_real_value.first_valid_index()]

        crypto_gambler_value = CryptoGambler(first_valid_index)
        crypto_gambler_mean = CryptoGambler(first_valid_index)
        crypto_gambler_var = CryptoGambler(first_valid_index)
        # crypto_gambler_varsign = CryptoGambler(first_valid_index)

        crypto_gambler_good_oracle = CryptoGambler(first_valid_index)
        crypto_gambler_baseline = CryptoGambler(first_valid_index)
        crypto_gambler_bad_oracle = CryptoGambler(first_valid_index)

        last_predict_value = 0.0
        last_real_value = 0.0
        last_predict_mean = 0.0
        i = 0
        for pred_dict, real_value in zip(predictions, l_real_value):
            i+=1
            if i%TIME_FRAME == 0: continue
            if last_predict_value == 0.0:
                last_predict_value = pred_dict['regressions_' + L_VALUE]
                last_predict_mean = pred_dict['regressions_' + L_MEAN]
                last_real_value = real_value
            else:
                optimal_predict = real_value - last_real_value
                predicted_value_diff = pred_dict['regressions_' + L_VALUE] - last_predict_value
                predicted_mean_diff = pred_dict['regressions_' + L_MEAN] - last_predict_mean
                predicted_var = pred_dict['regressions_' + L_VAR]

                crypto_gambler_value.evaluate_new_sample(real_value, predicted_value_diff)
                crypto_gambler_mean.evaluate_new_sample(real_value, predicted_mean_diff)
                crypto_gambler_var.evaluate_new_sample(real_value, predicted_var)

                crypto_gambler_good_oracle.evaluate_new_sample(last_real_value, optimal_predict)
                crypto_gambler_baseline.evaluate_new_sample(real_value, optimal_predict)
                crypto_gambler_bad_oracle.evaluate_new_sample(last_real_value, -optimal_predict)

                last_predict_value = pred_dict['regressions_' + L_VALUE]
                last_predict_mean = pred_dict['regressions_' + L_MEAN]
                last_real_value = real_value

        labels_shape = l_real_value[test_labels.shape[0]]
        Logger.bold("Bad Oracle Gambler:")
        crypto_gambler_bad_oracle.get_evaluation_results(labels_shape)
        Logger.bold("Baseline Gambler:")
        crypto_gambler_baseline.get_evaluation_results(labels_shape)
        Logger.bold("Eval Open Gambler:")
        crypto_gambler_value.get_evaluation_results(labels_shape)
        Logger.bold("Eval Mean Gambler:")
        crypto_gambler_mean.get_evaluation_results(labels_shape)
        Logger.bold("Eval Var Gambler:")
        crypto_gambler_var.get_evaluation_results(labels_shape)
        Logger.bold("Good Oracle Gambler:")
        crypto_gambler_good_oracle.get_evaluation_results(labels_shape)

    def evaluate(self, classifier, params, features, labels, visualize=False, mode='Train'):
        eval_result = classifier.evaluate(
            input_fn=lambda: DataConverter.eval_input_fn(features, labels, 1))
        for task_name in params['tasks'].keys():
            if params['tasks'][task_name].weight != 0:

                if task_name == 'l_open':
                    print('\n' + mode + ' set open mse: {mse_l_open:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_at_0':
                    print('\n' + mode + ' set  0 mse: {mse_l_open_at_0:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_at_1':
                    print('\n' + mode + ' set +1 mse: {mse_l_open_at_1:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_at_2':
                    print('\n' + mode + ' set +2 mse: {mse_l_open_at_2:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_at_3':
                    print('\n' + mode + ' set +3 mse: {mse_l_open_at_3:0.8f}\n'.format(**eval_result))

                if task_name == 'l_open_mean':
                    print('\n' + mode + ' set mean mse: {mse_l_open_mean:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_mean_at_3':
                    print('\n' + mode + ' set mean +3 mse: {mse_l_open_mean_at_3:0.8f}\n'.format(**eval_result))

                if task_name == 'l_open_var':
                    print('\n' + mode + ' set var mse: {mse_l_open_var:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_var_at_1':
                    print('\n' + mode + ' set var +1 mse: {mse_l_open_var_at_1:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_var_at_2':
                    print('\n' + mode + ' set var +2 mse: {mse_l_open_var_at_2:0.8f}\n'.format(**eval_result))
                if task_name == 'l_open_var_at_3':
                    print('\n' + mode + ' set var +3 mse: {mse_l_open_var_at_3:0.8f}\n'.format(**eval_result))

                if task_name == 'l_open_varsign':
                    print('\n' + mode + ' set accuracy varsign: {accuracy_l_open_varsign:0.3f}\n'.format(**eval_result))
                if task_name == 'l_open_varsign_at_1':
                    print('\n' + mode + ' set accuracy varsign +1: {accuracy_l_open_varsign_at_1:0.3f}\n'.format(**eval_result))
                if task_name == 'l_open_varsign_at_2':
                    print('\n' + mode + ' set accuracy varsign +2: {accuracy_l_open_varsign_at_2:0.3f}\n'.format(**eval_result))
                if task_name == 'l_open_varsign_at_3':
                    print('\n' + mode + ' set accuracy varsign +3: {accuracy_l_open_varsign_at_3:0.3f}\n'.format(**eval_result))

        if visualize:
            predictions = classifier.predict(
                input_fn=lambda: DataConverter.eval_input_fn(features, labels=None, batch_size=1))
            self.show_prediction_graph(predictions, labels, params)

    def show_prediction_graph(self, predictions, labels, params):

        lst_predict = {}
        lst_real = []
        lst_expect = []
        lst_var = []

        for task_name in params['tasks'].keys():
            if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                lst_predict[task_name] = []

        for pred_dict, expect, real, var in zip(predictions, labels[L_VALUE],
                                                labels['l_open_at_0'], labels[L_VAR]):
            for task_name in params['tasks'].keys():
                if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                    lst_predict[task_name].append(pred_dict['regressions_' + task_name])
            lst_expect.append(expect)
            lst_real.append(real)
            lst_var.append(var)

        for task_name in params['tasks'].keys():
            if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                plt.plot(lst_predict[task_name], label="predict_" + task_name)

        plt.plot(lst_expect, label="expect_value")
        plt.plot(lst_real, label="current")
        plt.plot(lst_var, label="expect_var")
        plt.legend(loc=0)
        plt.show()
