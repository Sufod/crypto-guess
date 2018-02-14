import tensorflow as tf

from tensorflow.contrib import layers


class CryptoModel:
    def model_fn(self, features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        initializer = tf.random_uniform_initializer(-params['init_scale'], params['init_scale'])

        #sequence_input = layers.sequence_input_from_feature_columns(
        #    columns_to_tensors=features,
        #    feature_columns=params['feature_columns'])

        # Input Layer
        net_in = tf.feature_column.input_layer(features, params['feature_columns'])

        # Hidden Layer
        for units, activ, drop_rate in zip(params['hidden_units'], params['hidden_activations'], params['dropout_rate']):
            net_in = tf.layers.dense(net_in, units=units, activation=activ, kernel_initializer=initializer)
            if drop_rate > 0.0:
                net_in = tf.layers.dropout(net_in, rate=drop_rate, training=is_training)

        # Output Layer
        outputs = {}
        predictions = {}
        metrics = {}
        loss = None
        for label_name in params['task_params'].keys():
            label_params = params['task_params'][label_name]

            if label_params['nb_classes'] == 1:
                # Compute regression predictions.
                if label_params['output_units'] is not None:
                    out = tf.identity(net_in)
                    for units, activ in zip(label_params['output_units'], label_params['output_activations']):
                        out = tf.layers.dense(out, units=units, activation=activ, kernel_initializer=initializer)
                    outputs[label_name] = tf.reshape(tf.layers.dense(out, 1, activation=None,
                                                                     kernel_initializer=initializer), [-1])
                else:
                    outputs[label_name] = tf.reshape(tf.layers.dense(net_in, 1, activation=None,
                                                                   kernel_initializer=initializer), [-1])
                # Compute predicts.
                predictions['regressions_' + label_name] = outputs[label_name]
            else:
                # Compute classification predictions.
                if label_params['output_units'] is not None:
                    out = tf.identity(net_in)
                    for units, activ in zip(label_params['output_units'], label_params['output_activations']):
                        out = tf.layers.dense(out, units=units, activation=activ, kernel_initializer=initializer)
                    outputs[label_name] = tf.layers.dense(out, label_params['nb_classes'],
                                                          activation=None,kernel_initializer=initializer)
                else:
                    outputs[label_name] = tf.layers.dense(net_in, label_params['nb_classes'],
                                                          activation=None, kernel_initializer=initializer)
                # Compute predicts.
                predicted_classes = tf.argmax(outputs[label_name], 1)
                predictions['class_ids_' + label_name] = predicted_classes[:, tf.newaxis]
                predictions['probabilities_' + label_name]=tf.nn.softmax(outputs[label_name])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        for label_name in params['task_params'].keys():
            label_params = params['task_params'][label_name]

            if label_params['nb_classes'] == 1:
                # Compute regress loss.
                if loss is not None:
                    loss = loss + label_params['weight']*tf.losses.mean_squared_error(labels=labels[label_name],
                                                               predictions=outputs[label_name])
                else:
                    loss = label_params['weight']*tf.losses.mean_squared_error(labels=labels[label_name],
                                                        predictions=outputs[label_name])
                # Compute evaluation metrics.
                mse = tf.metrics.mean_squared_error(labels=tf.cast(labels[label_name], tf.float32),
                                                    predictions=outputs[label_name],
                                                    name='mse_op_' + label_name)
                metrics['mse_' + label_name] = mse
                tf.summary.scalar('mse_' + label_name, mse[1])
            else:
                # Compute class loss.
                if loss is not None:
                    loss = loss + label_params['weight']*tf.losses.sparse_softmax_cross_entropy(labels=labels[label_name],
                                                                         logits=outputs[label_name])
                else:
                    loss = label_params['weight']*tf.losses.sparse_softmax_cross_entropy(labels=labels[label_name],
                                                                  logits=outputs[label_name])
                # Compute evaluation metrics.
                accuracy = tf.metrics.accuracy(labels=tf.cast(labels[label_name], tf.int32),
                                               predictions=tf.argmax(outputs[label_name], 1),
                                               name='acc_op_' + label_name)
                metrics['accuracy_' + label_name] = accuracy
                tf.summary.scalar('accuracy_' + label_name, accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        if params['optimizer'] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
        elif params['optimizer'] == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
