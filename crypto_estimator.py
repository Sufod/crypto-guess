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
from tasks.ClassificationTask import ClassificationTask
from tasks.RegressionTask import RegressionTask


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
    parser.add_argument('--num_steps', default=1000, type=int, help='number of recurrent steps')
    parser.add_argument('--train_steps', default=50000, type=int, help='number of training steps')
    args = parser.parse_args(argv[1:])

    tasks_list = [
        ClassificationTask("l_variation_sign", weight=0),
        RegressionTask("l_price_at_1")
    ]

    tasks = {}
    for task in tasks_list:
        tasks[task.name] = task

    params = {
        'optimizer': "Adagrad",
        'learning_rate': 0.01,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'train_steps': args.train_steps,
        'init_scale': 0.01,
        'hidden_units': [128, 64, 32],
        'hidden_activations': [tf.nn.relu, tf.nn.relu, tf.nn.relu],
        'dropout_rate': [0.0, 0.0, 0.0],
        'tasks': tasks
    }
    crypto_model=CryptoModel()
    crypto_brain=CryptoBrain()
    crypto_brain.run(crypto_model, params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
