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
from misc.CorpusUtils import CorpusUtils
from processors.FeaturesProcessor import FeaturesProcessor
from processors.FeaturesExtractor import FeaturesExtractor
from processors.CryptoLabelsExtractor import CryptoLabelsExtractor
from features.CorpusFeature import CorpusFeature
from features.Feature import Feature
from models.MultiLayerPerceptron import MultiLayerPerceptron
from tasks.ClassificationTask import ClassificationTask
from tasks.RegressionTask import RegressionTask


def main(argv):
    labels_extractor = CryptoLabelsExtractor()
    features_extractor = FeaturesExtractor()
    features_preprocessor = FeaturesProcessor()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--num_steps', default=100, type=int, help='number of recurrent steps')
    parser.add_argument('--train_steps', default=50000, type=int, help='number of training steps')
    args = parser.parse_args(argv[1:])

    params = {
        'corpora': {
            'train_corpus':'corpusmonnaies/BTC-train.csv',
            'dev_corpus':'corpusmonnaies/BTC-dev.csv',
            'test_corpus':'corpusmonnaies/BTC-test.csv',
        },
        'optimizer': "Adam",
        'learning_rate': 0.0005,
        'max_grad_norm': 0.1,
        'init_scale': 0.5,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'train_steps': args.train_steps,
        'supervision_steps': 1,
        'hidden_units': [64, 32],
        'hidden_activations': [None, None],
        'dropout_rate': [0.0, 0.0, 0.0],
        'tasks': [
            ClassificationTask(
                name="l_variation_sign",
                output_units=None,
                output_activations=None,
                weight=0,
                nb_classes=2,
                generate_method=lambda x: labels_extractor.compute_variation_sign(x)
            ),

            RegressionTask(
                name="l_price_at_0",
                output_units=None,
                output_activations=[None],
                weight=20,
                generate_method=lambda x: labels_extractor.compute_next_price_at(0, x)
            ),

            RegressionTask(
                name="l_real_price",
                output_units=None,
                output_activations=[None],
                weight=0,
                generate_method=lambda x: labels_extractor.compute_next_price_at(0, x)
            ),

            RegressionTask(
                name="l_price_at_1",
                output_units=[32, 16],
                output_activations=[None, None],
                weight=2,
                generate_method=lambda x: labels_extractor.compute_next_price_at(1, x)
            ),

            RegressionTask(
                name="l_price_at_2",
                output_units=[32, 16],
                output_activations=[None, None],
                weight=0,
                generate_method=lambda x: labels_extractor.compute_next_price_at(2, x)
            ),
        ],
        "features": [
            CorpusFeature(name='high'),
            CorpusFeature(name='low'),
            CorpusFeature(name='open'),
            CorpusFeature(name='volumefrom'),
            CorpusFeature(name='volumeto'),
            CorpusFeature(name='close'),
            Feature(
                name='high_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('high', -1, x)),
            Feature(
                name='low_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('low', -1, x)),
            Feature(
                name='open_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('open', -1, x)),
            Feature(
                name='volumefrom_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('volumefrom', -1, x)),
            Feature(
                name='volumeto_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('volumeto', -1, x)),
            Feature(
                name='close_at_-1',
                generate_method=lambda x: features_extractor.compute_feature_at('close', -1, x)),
            Feature(
                name='open_var_at_-1',
                generate_method=lambda x: features_extractor.compute_variation_feature('open', -1, x)),
            Feature(
                name='close_var_at_-1',
                generate_method=lambda x: features_extractor.compute_variation_feature('close', -1, x)),
            Feature(
                name='volume_mul',
                generate_method=lambda x: features_extractor.compute_arithmetic_feature('volumeto', 'mul', 'volumefrom', x)),
            Feature(
                name='volume_div',
                generate_method=lambda x: features_extractor.compute_arithmetic_feature('volumeto', 'div', 'volumefrom',x))
        ]
    }

    #CorpusUtils.produce_train_dev_test_from_full_corpus("corpusmonnaies/BTC-latest.csv")

    crypto_model = MultiLayerPerceptron()
    crypto_brain = CryptoBrain()
    crypto_brain.run(crypto_model, params)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
