import pandas as pd

from controllers.CryptoDataConverter import CryptoDataConverter
from controllers.CryptoDataLoader import CryptoDataLoader

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator
from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops

from tasks.RegressionTask import RegressionTask


class CryptoBrain:
    data_loader = CryptoDataLoader()

    def run(self, model, params):
        # Fetch the data
        corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-train.csv")
        data_converter = CryptoDataConverter(corpus)

        features, labels = data_converter.generate_features_and_labels(params)

        # Feature columns describe how to use the input.
        model_feature_columns = []
        for feature in features.keys():
            model_feature_columns.append(tf.feature_column.numeric_column(key=feature))
        params['feature_columns'] = model_feature_columns

        classifier = tf.estimator.Estimator(model_fn=model.model_fn, params=params)

        # Train the model.
        classifier.train(
            input_fn=lambda: data_converter.train_input_fn(features, labels, params['batch_size']),
            steps=params['train_steps'])
        predictions = classifier.predict(
            input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        self.show_prediction_graph(predictions, labels, params)

        # # Re-Train the model.
        # params['learning_rate'] = 0.001
        # params['batch_size'] = 1
        # params['train_steps'] = 100
        # classifier.train(
        #     input_fn=lambda: data_converter.train_input_fn(features, labels, params['batch_size']),
        #     steps=params['train_steps'])
        # predictions = classifier.predict(
        #     input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        # self.show_prediction_graph(predictions, labels, params)

        # Evaluate the model.
        corpus = self.data_loader.load_crypto_crawler_data("corpusmonnaies/BTC-test.csv")
        data_converter = CryptoDataConverter(corpus)
        features, labels = data_converter.generate_features_and_labels(params)
        features = pd.concat([features, corpus], axis=1)
        eval_result = classifier.evaluate(
            input_fn=lambda: data_converter.eval_input_fn(features, labels, 1))
        for task_name in params['tasks'].keys():
            if params['tasks'][task_name].weight != 0:
                if task_name == 'l_price_at_0':
                    print('\nTest set  0 mse: {mse_l_price_at_0:0.6f}\n'.format(**eval_result))
                if task_name == 'l_price_at_1':
                    print('\nTest set +1 mse: {mse_l_price_at_1:0.6f}\n'.format(**eval_result))
                if task_name == 'l_price_at_2':
                    print('\nTest set +2 mse: {mse_l_price_at_2:0.6f}\n'.format(**eval_result))
                if task_name == 'l_variation_sign':
                    print('\nTest set accuracy variation sign: {accuracy_l_variation_sign:0.3f}\n'.format(**eval_result))

        # Visualize predictions.
        predictions = classifier.predict(
            input_fn=lambda: data_converter.eval_input_fn(features, labels=None, batch_size=1))
        self.show_prediction_graph(predictions, labels, params)

        # predictions = classifier.predict(
        #    input_fn=lambda: self.crypto_data_converter.eval_input_fn(features,labels=None,batch_size=1))
        # for pred_dict, expect_class, expect_regress1, expect_regress2 in zip(predictions, labels[0], labels[1], labels[2]):
        #    template = ('\nPrediction is "{}" / "{:.2f}" ({:.1f}%), expected "{}" / "{}"')
        #    class_id = pred_dict['class_ids_0'][0]
        #    probability = pred_dict['probabilities_0'][class_id]
        #    print(template.format(class_id, pred_dict['regressions_1'], pred_dict['regressions_2'],
        #                          100 * probability, expect_class, expect_regress1, expect_regress2))

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
                    lst_predict[task_name].append(pred_dict['regressions_'+task_name])
            lst_expect.append(expect)
            lst_real.append(real)

        for task_name in params['tasks'].keys():
            if isinstance(params['tasks'][task_name], RegressionTask) and params['tasks'][task_name].weight != 0:
                plt.plot(lst_predict[task_name], label="predict_"+task_name)

        plt.plot(lst_expect, label="real+1")
        plt.plot(lst_real, label="real")
        plt.legend(loc=0)
        plt.show()
