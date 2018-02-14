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
from models.CryptoModel import CryptoModel


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--num_steps', default=100, type=int, help='number of recurrent steps')
    parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
    args = parser.parse_args(argv[1:])

    params = {
        'optimizer': "SGD",
        'learning_rate': 0.01,
        'batch_size': 100,
        'init_scale': 0.01,
        'hidden_units': [128, 64, 32],
        'hidden_activations': [tf.nn.relu, tf.nn.relu, tf.nn.relu],
        'dropout_rate': [0.0, 0.0, 0.0],
        'task_params':
            {
            'l_variation_sign':
                {
                'output_units': None,
                'output_activations': [None],
                'nb_classes': 4,
                'weight': 0
                },
            'l_price_at_1':
                {
                'output_units': [16, 8],
                'output_activations': [tf.nn.relu, tf.nn.relu],
                'nb_classes': 1,
                'weight': 1
                },
            'l_price_at_2':
                {
                'output_units': [16, 8],
                'output_activations': [tf.nn.relu, tf.nn.relu],
                'nb_classes': 1,
                'weight': 1
                },
            'l_price_at_0':
                {
                'output_units': None,
                'output_activations': [None],
                'nb_classes': 1,
                'weight': 0
                }
            }
    }

    model = CryptoModel()
    network = CryptoBrain()
    network.run(args.batch_size, args.num_steps, args.train_steps, model, params)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
