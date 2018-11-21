import abc
import numpy as np
import six
import os
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class NeuralNetwork(object):
    """Abstract base class for Neural Network used in 
    policy-value net.

    Details can be found in https://www.nature.com/articles/nature24270
    'Mastering the game of Go without human knowledge'
    """
    @abc.abstractmethod
    def policyValueFunc(self, board):
        pass

    @abc.abstractmethod
    def trainStep(self, state_batch, mcts_probs_batch, winner_batch, lr):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def restore(self, path):
        pass
    
    @abc.abstractproperty
    def width(self):
        pass

    @abc.abstractproperty
    def height(self):
        pass


class SimpleCNN(NeuralNetwork):
    def __init__(self, height, width, model_file=None, norm_weight=1e-4):
        self.board_width = width
        self.board_height = height

        # Define the neural network
        with tf.variable_scope("SimpleCNN"):
            # input placeholders

            # input states placeholder, 4 channels are:
            #   board_state[0]: current board state with only current player's stone
            #   board_state[1]: current board state with only opponent's stones
            #   board_state[2]: only one stone, indicate the last move(opponent made this move).
            #   board_state[3]: indicate the player to play, 0 for white, 1 for black
            self.raw_input_states = tf.placeholder(
                tf.float32, shape=[None, 4, height, width])

            # label contains the result of game
            self.value_labels = tf.placeholder(tf.float32, shape=[None, 1])

            # label contains the probability vector from MCTS for each step of game
            self.mcts_probs_labels = tf.placeholder(
                tf.float32, shape=[None, height*width])

            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            # tensorflow like input with format [N,H,W,C]
            self.input_states = tf.transpose(self.raw_input_states, [0, 2, 3, 1])

            # Shared Layers
            with tf.variable_scope("shared_layers"):
                self.conv1 = tf.layers.conv2d(inputs=self.input_states,
                                              filters=32, kernel_size=3,
                                              padding="same", activation=tf.nn.relu,
                                              name="conv1")
                self.batchnorm1 = tf.layers.batch_normalization(self.conv1, training=self.is_training)
                self.conv2 = tf.layers.conv2d(inputs=self.batchnorm1, filters=64,
                                              kernel_size=3, padding="same",
                                              activation=tf.nn.relu, name="conv2")
                self.batchnorm2 = tf.layers.batch_normalization(self.conv2, training=self.is_training)
                self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                              kernel_size=3, padding="same",
                                              activation=tf.nn.relu, name="conv3")
                self.batchnorm3 = tf.layers.batch_normalization(self.conv3, training=self.is_training)
            # Action net layers
            with tf.variable_scope("action_layers"):
                self.action_conv = tf.layers.conv2d(inputs=self.batchnorm3, filters=8,
                                                    kernel_size=1, padding="same",
                                                    activation=tf.nn.relu, name="action_conv")
                self.action_conv_flat = tf.reshape(
                    self.action_conv, [-1, 8 * height * width])
                self.action_out = tf.layers.dense(inputs=self.action_conv_flat,
                                                  units=height*width,
                                                  activation=tf.nn.softmax,
                                                  name="action_out")
                self.action_out_log = tf.log(self.action_out)

            # Value net layers
            with tf.variable_scope("value_layers"):
                self.value_conv = tf.layers.conv2d(inputs=self.batchnorm3, filters=2,
                                                   kernel_size=1, padding="same",
                                                   activation=tf.nn.relu, name="value_conv")
                self.value_conv_flat = tf.reshape(
                    self.value_conv, [-1, 2 * height * width]
                )
                self.value_fc = tf.layers.dense(inputs=self.value_conv_flat, units=64,
                                                activation=tf.nn.relu, name="value_fc")
                self.value_out = tf.layers.dense(inputs=self.value_fc, units=1,
                                                 activation=tf.nn.tanh, name="value_out")

            # losses
            self.value_loss = tf.losses.mean_squared_error(
                self.value_labels, self.value_out)
            self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(
                self.mcts_probs_labels, self.action_out_log), 1)))

            trainable_vars = tf.trainable_variables()
            self.l2_norm_weight = norm_weight
            l2_norm = norm_weight * tf.add_n(
                [tf.nn.l2_loss(v) for v in trainable_vars if ('bias' not in v.name.lower() and 
                                                              'moving' not in v.name.lower())])

            self.loss = self.value_loss + self.policy_loss + l2_norm
            self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(self.action_out * tf.log(self.action_out), -1)
            ))

            # train op part
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)
            self.global_step = tf.get_variable("global_step", initializer=0, trainable=False)
            self.step_add_op = self.global_step + 1

            # session
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            # Saver
            self.saver = tf.train.Saver()
            if model_file is not None:
                self.restore(model_file)

    def restore(self, model_path):
        dir_path = os.path.dirname(model_path)
        self.saver.restore(self.session, tf.train.latest_checkpoint(dir_path))

    def save(self, model_path):
        global_step = self.getGlobalStep()
        dir_path = os.path.dirname(model_path)
        if not tf.gfile.Exists(dir_path):
            tf.gfile.MakeDirs(dir_path)
        self.saver.save(self.session, model_path, global_step=global_step)

    def getPolicyValue(self, state_batch):
        act_prob, value = self.session.run(
            [self.action_out, self.value_out],
            feed_dict={self.raw_input_states: state_batch, self.is_training: False}
        )
        return act_prob, value

    def policyValueFunc(self, board):
        """The Policy-value function.

        This function takes a board state and return evaluation value 
        and next_action probability vector.
        """
        valid_positions = board.availables
        current_state = np.ascontiguousarray(board.currentState().reshape(
            -1, 4, self.board_height, self.board_width))
        policy_vec, value = self.getPolicyValue(current_state)
        # 0 because getPolicyValue takes batch of data
        policy_vec = zip(valid_positions, policy_vec[0][valid_positions])
        return policy_vec, value

    def trainStep(self, state_batch, mcts_probs_batch, winner_batch, lr):
        """Perform single training step.

        Args:
            state_batch: A numpy array of board state used as the training data.
            mcts_probs_batch: A numpy array of action probability vectors 
                used as training label.
            winner_batch: A numpy array of game result used as training label.
            lr: learning rate.
        """
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, _, _, entropy = self.session.run(
            [self.loss, self.train_op, self.step_add_op, self.entropy],
            feed_dict={self.raw_input_states: state_batch,
                       self.mcts_probs_labels: mcts_probs_batch,
                       self.value_labels: winner_batch,
                       self.learning_rate: lr,
                       self.is_training: True})
        return loss, entropy
    
    def getGlobalStep(self):
        global_step = self.session.run(self.global_step)
        return global_step

    @property
    def width(self):
        return self.board_width

    @property
    def height(self):
        return self.board_height
