#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from controllers.CryptoBrain import CryptoBrain
from features.SymbolicFeature import SymbolicFeature

from processors.FeaturesExtractor import FeaturesExtractor
from features.NumericFeature import NumericFeature
from models.MultiLayerPerceptron import MultiLayerPerceptron
from processors.FeaturesProcessor import FeaturesProcessor
from tasks.ClassificationTask import ClassificationTask
from tasks.RegressionTask import RegressionTask


def main(argv):
    features_extractor = FeaturesExtractor()
    features_processor = FeaturesProcessor()


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--num_steps', default=100, type=int, help='number of recurrent steps')
    parser.add_argument('--train_steps', default=100000, type=int, help='number of training steps')
    args = parser.parse_args(argv[1:])

    params = {
        'corpora': {
            'autogen': 'corpusmonnaies/BTC-latest.csv',
            # Below values are ignored if autogen is set
            'train': 'corpusmonnaies/BTC-train.csv',
            'dev': 'corpusmonnaies/BTC-dev.csv',
            'test': 'corpusmonnaies/BTC-test.csv'
        },
        'optimizer': "Adam",
        'learning_rate': 0.0005,
        'max_grad_norm': 0.1,
        'init_scale': 0.5,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'train_steps': args.train_steps,
        'supervision_steps': 10,
        'hidden_units': [64, 32],
        'hidden_activations': [None, None],
        'dropout_rate': [0.0, 0.0, 0.0],
        'tasks': [
            RegressionTask(
                name="l_current_open",
                output_units=None,
                output_activations=[None],
                weight=0,
                normalize=False,
                generate_method=lambda x: features_extractor.compute_feature_at('open', 0, x)
            ),

            RegressionTask(
                name="l_open",
                output_units=[32, 16],
                output_activations=[None, None],
                weight=1,
                generate_method=lambda x: features_extractor.compute_feature_at('open', 3, x)
            ),

            RegressionTask(
                name="l_open_mean",
                output_units=[32, 16],
                output_activations=[None, None],
                weight=1,
                generate_method=lambda x: features_extractor.compute_mean_on('open', 3, x)
            ),

            RegressionTask(
                name="l_open_var",
                output_units=[32, 16],
                output_activations=[None, None],
                weight=1,
                normalize=False,
                normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
                generate_method=lambda x: features_extractor.compute_variation_at('open', 3, x)
            ),

            ClassificationTask(
                name="l_open_varsign",
                output_units=None,
                output_activations=[None, None],
                weight=1,
                nb_classes=4,
                generate_method=lambda x: features_extractor.compute_variation_sign_at('open', 3, x)
            ),

            RegressionTask(
                name="l_open_at_0",
                output_units=None,
                output_activations=[None],
                weight=1,
                generate_method=lambda x: features_extractor.compute_feature_at('open', 0, x)
            )

            # RegressionTask(
            #     name="l_open_at_1",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     generate_method=lambda x: features_extractor.compute_feature_at('open', 1, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_at_2",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     generate_method=lambda x: features_extractor.compute_feature_at('open', 2, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_at_3",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     generate_method=lambda x: features_extractor.compute_feature_at('open', 3, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_mean_at_3",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     generate_method=lambda x: features_extractor.compute_mean_on('open', 3, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_var_at_1",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     normalize=False,
            #     normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
            #     generate_method=lambda x: features_extractor.compute_variation_at('open', 1, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_var_at_2",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     normalize=False,
            #     normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
            #     generate_method=lambda x: features_extractor.compute_variation_at('open', 1, x)
            # ),
            #
            # RegressionTask(
            #     name="l_open_var_at_3",
            #     output_units=[32, 16],
            #     output_activations=[None, None],
            #     weight=0,
            #     normalize=False,
            #     normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
            #     generate_method=lambda x: features_extractor.compute_variation_at('open', 3, x)
            # ),
            #
            # ClassificationTask(
            #     name="l_open_varsign_at_1",
            #     output_units=None,
            #     output_activations=[None, None],
            #     weight=0,
            #     nb_classes=4,
            #     generate_method=lambda x: features_extractor.compute_variation_sign_at('open', 3, x)
            # ),
            #
            # ClassificationTask(
            #     name="l_open_varsign_at_2",
            #     output_units=None,
            #     output_activations=[None, None],
            #     weight=0,
            #     nb_classes=4,
            #     generate_method=lambda x: features_extractor.compute_variation_sign_at('open', 2, x)
            # ),
            #
            # ClassificationTask(
            #     name="l_open_varsign_at_3",
            #     output_units=None,
            #     output_activations=[None, None],
            #     weight=0,
            #     nb_classes=4,
            #     generate_method=lambda x: features_extractor.compute_variation_sign_at('open', 3, x)
            # )
        ],
        "corpus_features": [
            NumericFeature(name='high'),
            NumericFeature(name='low'),
            NumericFeature(name='open'),
            NumericFeature(name='close'),
            NumericFeature(name='volumefrom'),
            NumericFeature(name='volumeto'),
        ],
        "features": [
            SymbolicFeature(
                name='open_varsign_at_-1',
                vocabulary=(0, 1, 2, 3),
                embedding_units=4,
                normalize=False,
                generate_method=lambda x: features_extractor.compute_variation_sign_at('open', -1, x)),
            NumericFeature(
                name='open_var_at_-1',
                normalize=False,
                normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
                generate_method=lambda x: features_extractor.compute_variation_at('open', -1, x)),
            NumericFeature(
                name='open_var_at_-2',
                normalize=False,
                normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
                generate_method=lambda x: features_extractor.compute_variation_at('open', -2, x)),
            NumericFeature(
                name='close_var_at_-1',
                normalize=False,
                normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
                generate_method=lambda x: features_extractor.compute_variation_at('close', -1, x)),
            NumericFeature(
                name='close_var_at_-2',
                normalize=False,
                normalize_inflow=lambda x: features_processor.normalize_series(-1.0, 1.0, x, centered=True),
                generate_method=lambda x: features_extractor.compute_variation_at('close', -2, x)),
            NumericFeature(
                name='open_mean_at_-5',
                generate_method=lambda x: features_extractor.compute_mean_on('open', -5, x)),
            NumericFeature(
                name='high_max_at_-5',
                generate_method=lambda x: features_extractor.compute_max_on('high', -5, x)),
            NumericFeature(
                name='low_min_at_-5',
                generate_method=lambda x: features_extractor.compute_min_on('low', -5, x)),
            NumericFeature(
                name='close_mean_at_-5',
                generate_method=lambda x: features_extractor.compute_mean_on('close', -5, x)),
            NumericFeature(
                name='volumeto_mean_at_-5',
                generate_method=lambda x: features_extractor.compute_mean_on('volumeto', -5, x)),
            NumericFeature(
                name='volumefrom_mean_at_-5',
                generate_method=lambda x: features_extractor.compute_mean_on('volumefrom', -5, x)),
            NumericFeature(
                name='volume_diff',
                generate_method=lambda x: features_extractor.compute_arithmetic_feature('volumeto', 'sub', 'volumefrom',
                                                                                        x))
        ],
        "post_process_features": [
            lambda x: features_processor.create_context_window(3, x)
        ]
    }

    crypto_model = MultiLayerPerceptron()
    crypto_brain = CryptoBrain()
    crypto_brain.run(crypto_model, params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
