import tensorflow as tf

from tensorflow.contrib import layers

from tasks.ClassificationTask import ClassificationTask
from tasks.RegressionTask import RegressionTask


class CryptoModel:
    def model_fn(self, features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        initializer = tf.random_uniform_initializer(-params['init_scale'], params['init_scale'])

        # sequence_input = layers.sequence_input_from_feature_columns(
        #    columns_to_tensors=features,
        #    feature_columns=params['feature_columns'])

        # Input Layer
        net_in = tf.feature_column.input_layer(features, params['feature_columns'])

        # Hidden Layer
        for units, activ, drop_rate in zip(params['hidden_units'], params['hidden_activations'],
                                           params['dropout_rate']):
            net_in = tf.layers.dense(net_in, units=units, activation=activ, kernel_initializer=initializer)
            if drop_rate > 0.0:
                net_in = tf.layers.dropout(net_in, rate=drop_rate, training=is_training)

        # Output Layer
        outputs = {}
        predictions = {}
        metrics = {}
        loss = None
        for task_name in params['tasks'].keys():
            current_task = params['tasks'][task_name]

            if isinstance(current_task, RegressionTask):

                # Compute regression predictions.
                if current_task.output_units is not None:
                    out = tf.identity(net_in)
                    for units, activ in zip(current_task.output_units, current_task.output_activations):
                        out = tf.layers.dense(out, units=units, activation=activ, kernel_initializer=initializer)
                    outputs[task_name] = tf.reshape(tf.layers.dense(out, 1, activation=None,
                                                                    kernel_initializer=initializer), [-1])
                else:
                    outputs[task_name] = tf.reshape(tf.layers.dense(net_in, 1, activation=None,
                                                                    kernel_initializer=initializer), [-1])
                # Compute predicts.
                predictions['regressions_' + task_name] = outputs[task_name]

            elif isinstance(current_task, ClassificationTask):

                # Compute classification predictions.
                if current_task.output_units is not None:
                    out = tf.identity(net_in)
                    for units, activ in zip(current_task.output_units, current_task.output_activations):
                        out = tf.layers.dense(out, units=units, activation=activ, kernel_initializer=initializer)
                    outputs[task_name] = tf.layers.dense(out, current_task.nb_classes,
                                                         activation=None, kernel_initializer=initializer)
                else:
                    outputs[task_name] = tf.layers.dense(net_in, current_task.nb_classes,
                                                         activation=None, kernel_initializer=initializer)
                # Compute predicts.
                predicted_classes = tf.argmax(outputs[task_name], 1)
                predictions['class_ids_' + task_name] = predicted_classes[:, tf.newaxis]
                predictions['probabilities_' + task_name] = tf.nn.softmax(outputs[task_name])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        for task_name in params['tasks'].keys():
            current_task = params['tasks'][task_name]

            if isinstance(current_task, RegressionTask):
                # Compute regress loss.
                if loss is not None:
                    loss = loss + current_task.weight * tf.losses.mean_squared_error(labels=labels[task_name],
                                                                                     predictions=outputs[task_name])
                else:
                    loss = current_task.weight * tf.losses.mean_squared_error(labels=labels[task_name],
                                                                              predictions=outputs[task_name])
                # Compute evaluation metrics.
                mse = tf.metrics.mean_squared_error(labels=tf.cast(labels[task_name], tf.float32),
                                                    predictions=outputs[task_name],
                                                    name='mse_op_' + task_name)
                metrics['mse_' + task_name] = mse
                tf.summary.scalar('mse_' + task_name, mse[1])

            elif isinstance(current_task, ClassificationTask):
                # Compute class loss.
                if loss is not None:
                    loss = loss + current_task.weight * tf.losses.sparse_softmax_cross_entropy(labels=labels[task_name],
                                                                                               logits=outputs[
                                                                                                   task_name])
                else:
                    loss = current_task.weight * tf.losses.sparse_softmax_cross_entropy(labels=labels[task_name],
                                                                                        logits=outputs[task_name])
                # Compute evaluation metrics.
                accuracy = tf.metrics.accuracy(labels=tf.cast(labels[task_name], tf.int32),
                                               predictions=tf.argmax(outputs[task_name], 1),
                                               name='acc_op_' + task_name)
                metrics['accuracy_' + task_name] = accuracy
                tf.summary.scalar('accuracy_' + task_name, accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        if params['optimizer'] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
        elif params['optimizer'] == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        elif params['optimizer'] == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

        print(optimizer)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
