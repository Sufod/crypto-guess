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
from controllers.CryptoFeaturesPreprocessor import CryptoFeaturesPreprocessor
from extractors.CryptoFeaturesExtractor import CryptoFeaturesExtractor
from extractors.CryptoLabelsExtractor import CryptoLabelsExtractor
from models.CryptoModel import CryptoModel
from tasks.ClassificationTask import ClassificationTask
from tasks.RegressionTask import RegressionTask


def main(argv):
    labels_extractor = CryptoLabelsExtractor()
    features_extractor = CryptoFeaturesExtractor()
    features_preprocessor = CryptoFeaturesPreprocessor()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--num_steps', default=1000, type=int, help='number of recurrent steps')
    parser.add_argument('--train_steps', default=100000, type=int, help='number of training steps')
    args = parser.parse_args(argv[1:])

    params = {
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
        'tasks': get_tasks_from_tasks_list([
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
        ])
    }
    crypto_model = CryptoModel()
    crypto_brain = CryptoBrain()
    crypto_brain.run(crypto_model, params)


def get_tasks_from_tasks_list(tasks_list):
    tasks = {}
    for task in tasks_list:
        tasks[task.name] = task
    return tasks


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
