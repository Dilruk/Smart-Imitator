import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from imblearn.over_sampling import SMOTE
import Utils.utils_main as utils
import Utils.settings as setting_4
import argparse
import os
import time
from datetime import timedelta
from datetime import datetime
import json
from sklearn.utils import shuffle
import pandas as pd
import copy
import Utils.utils_trajectories as TrajUtils


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Learn Optimal Policy.")
    parser.add_argument('--no_epochs_pretrain', type=int, nargs='?', default=2000, help='Number of pretrain epochs.')
    parser.add_argument('--no_epochs_train', type=int, nargs='?', default=5000, help='Number of train epochs.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--qnet_hidden_size', type=int, default=256, help='Policy hidden size.')
    parser.add_argument('--discriminator_hidden_size', nargs='?', default=8, help="Discriminator hidden size.")
    parser.add_argument('--gamma', nargs='?', default=0.99, help="Discount Factor.")
    parser.add_argument('--D_rewards_weight', nargs='?', default=10, help="Weight of Discriminator feedack in the rewards.")
    parser.add_argument('--target_replace_freq', nargs='?', default=50, help="Discount Factor.")
    parser.add_argument('--lr', nargs='?', default=0.00001, help="Policy Learning rate.")
    parser.add_argument('--lr_dA', nargs='?', default=0.0001, help="Discriminator_A Learning rate.")
    parser.add_argument('--lr_dC', nargs='?', default=0.001, help="Discriminator_C Learning rate.")
    parser.add_argument('--tau', nargs='?', default=0.0005, help="TAU, proportion of main network value used when updating target network.")
    parser.add_argument('--reward_Threshold_q', nargs='?', default=20.0, help="Reward Threshold for Q net.")
    parser.add_argument('--reward_Threshold_d', nargs='?', default=6.0, help="Reward Threshold for D.")
    parser.add_argument('--reg_lambda', nargs='?', default=2.0, help="Regularization Lambda.")
    parser.add_argument('--reg_lambda_d', nargs='?', default=0.01, help="Regularization Lambda.")
    parser.add_argument('--lambda_neg', nargs='?', default=1.0, help="contribution from negatives.")
    parser.add_argument('--lambda_r1', nargs='?', default=0.8, help="Scalar to maximize the smaller rewards getting from log operations.")
    parser.add_argument('--dropout', nargs='?', default=0.0, help="dropout for policy.")
    parser.add_argument('--dropout_d', nargs='?', default=0.0, help="dropout for D.")
    parser.add_argument('--exp_policy_split_frac', nargs='?', default=0.3, help="Exp to Policy Division.")
    parser.add_argument('--FINAL_REWARD_SCALE', nargs='?', default=15, help="Mortality reward/penalty volume.")
    parser.add_argument('--THRESHOLD_NONOP_PAIRS', nargs='?', default=0., help="THRESHOLD_NONOP_PAIRS.")
    parser.add_argument('--EPOCHS_PER_PLOT_DATA', nargs='?', default=2, help="Epochs per plot")
    parser.add_argument('--EPOCHS_PER_PLOT', nargs='?', default=2, help="Epochs per plot")
    return parser.parse_args()


class Qnetwork():
    def __init__(self, ob_size_, ac_size_, name, hidden_size_, reward_threshold, reg_lambda_, lr_):
        self.input_size = ob_size_
        self.num_actions = ac_size_
        self.name = name
        self.hidden_size = hidden_size_
        self.REWARD_THRESHOLD = reward_threshold
        self.reg_lambda = reg_lambda_
        self.mse = tf.keras.losses.MeanSquaredError()
        self.huber = tf.compat.v1.losses.huber_loss
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr_)
        self.state = tf.placeholder(tf.float32, shape=[None, self.input_size], name=str(name) + "_input_state")
        self.actions_onehot = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dropout_rate = tf.placeholder(tf.float32)
        self._create_q_graph(self.name)

    def _create_q_graph(self, name):
        with tf.variable_scope(name + '_q_network'):
            d_h1_ = layers.fully_connected(self.state, self.hidden_size, activation_fn=tf.nn.relu)
            q_out = layers.fully_connected(d_h1_, self.hidden_size//2, activation_fn=tf.keras.activations.linear)
            # Advantage and value streams
            # Advantage
            self.A_h1_ = layers.fully_connected(q_out, self.hidden_size // 4, activation_fn=tf.nn.relu)
            self.Advantage = layers.fully_connected(self.A_h1_, self.num_actions, activation_fn=None)

            # Value
            self.V_h1_ = layers.fully_connected(q_out, self.hidden_size // 4, activation_fn=tf.nn.relu)
            self.Value = layers.fully_connected(self.V_h1_, 1, activation_fn=None)

            # Then combine them together to get our final Q-values.
            self.q_outputs = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            self.predicted_best_action = tf.argmax(self.q_outputs, 1, name='predict')  # vector of length batch size

            # Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
            # select the Q values for the actions that would be selected
            self.Q = tf.reduce_sum(tf.multiply(self.q_outputs, self.actions_onehot), reduction_indices=1)  # batch size x 1 vector

            # regularisation penalises the network when it produces rewards that are above the
            # reward threshold, to ensure reasonable Q-value predictions
            reg_vector = tf.maximum(tf.abs(self.Q) - self.REWARD_THRESHOLD, 0)
            reg_term = tf.reduce_sum(reg_vector)

            self.mean_abs_error = self.mae(self.targetQ, self.Q)
            self.loss = self.huber(self.targetQ, self.Q) + self.reg_lambda * reg_term

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.optimizer.minimize(loss=self.loss, var_list=self.var_list)


class Policy:
    def __init__(self, sess, ob_size_, ac_size_, Q_MAIN_SCOPE, Q_TARGET_SCOPE, qnet_hidden_size_, reward_Threshold_q_, reg_lambda_, lr_, tau, gamma, D_reward_weight):
        self.sess = sess
        self.ob_size = ob_size_
        self.ac_size = ac_size_

        self.qmain_scope_name = Q_MAIN_SCOPE
        self.qtarget_scope_name = Q_TARGET_SCOPE
        self.tau = tau
        self.gamma = gamma
        self.D_reward_weight = D_reward_weight

        self.qnet_hidden_size = qnet_hidden_size_
        self.reward_Threshold_q = reward_Threshold_q_
        self.reg_lambda = reg_lambda_
        self.lr = lr_

        self.mainQN = Qnetwork(self.ob_size, self.ac_size, self.qmain_scope_name, args.qnet_hidden_size, args.reward_Threshold_q, args.reg_lambda, float(args.lr))
        self.targetQN = Qnetwork(self.ob_size, self.ac_size, self.qtarget_scope_name, args.qnet_hidden_size, args.reward_Threshold_q, args.reg_lambda, float(args.lr))

        self.main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.qmain_scope_name)
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.qtarget_scope_name)

        self.replace_target_op = [tf.assign(t, t+self.tau * (e-t)) for t, e in zip(self.target_vars, self.main_vars)]

    def update_mainQ_and_get_loss_with_DaDc_feedback(self, obs_batch, acs_batch, next_obs_batch, not_done_flags, dropout_rate, rewards, Da_feedback, Dc_feedback):
        """
        Update the MainQ network with discriminator feedback and calculate the loss.

        Parameters:
        - obs_batch: Batch of observations
        - acs_batch: Batch of actions taken
        - next_obs_batch: Batch of next observations
        - not_done_flags: Batch indicating if the episode has not ended
        - dropout_rate: Dropout rate for regularization
        - Da_feedback: Feedback from the adversarial discriminator
        - Dc_feedback: Feedback from the cooperative discriminator

        Returns:
        - loss: The loss after the update
        - error: The mean absolute error
        """

        # Get the indices of the best actions for the next states from the MainQ Network
        best_actions_next = self.sess.run(self.mainQN.predicted_best_action,
                                          feed_dict={self.mainQN.state: next_obs_batch,
                                                     self.mainQN.dropout_rate: dropout_rate})

        # Get the Q-values for these best actions from the TargetQ Network
        q_values_next_target = self.sess.run(self.targetQN.q_outputs,
                                             feed_dict={self.targetQN.state: next_obs_batch,
                                                        self.targetQN.dropout_rate: dropout_rate})

        # Extract the Q-values for the chosen actions
        double_q_values = q_values_next_target[np.arange(len(q_values_next_target)), best_actions_next]

        # Incorporate discriminator feedback into the target Q-value calculation
        adjusted_rewards = rewards + self.D_reward_weight * (Da_feedback + Dc_feedback)

        # Compute the target Q-values with adjusted rewards
        # target_q_values = adjusted_rewards + (self.gamma * double_q_values * not_done_flags)
        target_q_values = (self.gamma * double_q_values * not_done_flags)

        # Perform the update on the MainQ Network and get the loss and error
        _, loss, error = self.sess.run([self.mainQN.update_model, self.mainQN.loss, self.mainQN.mean_abs_error],
                                       feed_dict={self.mainQN.state: obs_batch,
                                                  self.mainQN.actions_onehot: acs_batch,
                                                  self.mainQN.targetQ: target_q_values,
                                                  self.mainQN.dropout_rate: dropout_rate})

        return loss, error

    def update_target(self):
        self.sess.run(self.replace_target_op)


# Adversarial Discriminator
class Discriminator_Adv:
    def __init__(self, sess, ob_shape, ac_shape, beta_, hidden_size, lr_d, lambda_r1_, lambda_neg_, reg_lambda_d_, name, REWARD_THRESHOLD_D):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.beta = beta_
        self.hidden_size = hidden_size
        self.lr_d = lr_d
        self.lambda_r1_ = lambda_r1_
        self.lambda_nonop = lambda_neg_
        self.reg_lambda_d = reg_lambda_d_
        self.name = name
        self.REWARD_THRESHOLD_D = REWARD_THRESHOLD_D
        self.optimizer_d = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr_d)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ob_ac = tf.placeholder(dtype=tf.float32, shape=[None, ob_shape[0] + ac_shape[0]])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None])
        self.dropout_rate_d = tf.placeholder(tf.float32)
        self._build_network(self.name)

    def _build_network(self, name):
        with tf.variable_scope(str(name) + '_discriminator'):
            d_h1_ = layers.fully_connected(self.ob_ac, self.hidden_size, activation_fn=tf.nn.leaky_relu)
            d_h1 = tf.nn.dropout(d_h1_, rate=self.dropout_rate_d)
            d_h2_ = layers.fully_connected(d_h1, int(self.hidden_size/4), activation_fn=tf.nn.leaky_relu)
            d_h2 = tf.nn.dropout(d_h2_, rate=self.dropout_rate_d)
            self.d_out_ = layers.fully_connected(d_h2, 1, activation_fn=tf.sigmoid)
            self.d_out = tf.squeeze(self.d_out_)
            # typical GAN loss computation
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            self.loss = self.bce_loss(y_true=self.labels, y_pred=self.d_out)
            self.mean_abs_error = self.mae(y_true=self.labels, y_pred=self.d_out)
            self.train_op = self.optimizer_d.minimize(loss=self.loss, var_list=self.var_list)

    def get_feedback(self, ob_ac_, dropout_):
        """
        Get feedback from the adversarial discriminator based on observation-action pairs.
        """
        feed_dict = {self.ob_ac: ob_ac_, self.dropout_rate_d: dropout_}
        d_out = self.sess.run(self.d_out, feed_dict=feed_dict)
        return d_out

    def update_n_getloss(self, all_ob_ac_, labels_, dropout_):
        feed_dict = {self.ob_ac: all_ob_ac_, self.labels: labels_, self.dropout_rate_d: dropout_}
        _, d_loss, error = self.sess.run((self.train_op, self.loss, self.mean_abs_error), feed_dict=feed_dict)
        return d_loss, error

# Cooperative Discriminator
class Discriminator_Cop:
    def __init__(self, sess, ob_shape, ac_shape, beta_, hidden_size, lr_d, lambda_r1_, lambda_neg_, reg_lambda_d_, name, REWARD_THRESHOLD_D):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.beta = beta_
        self.hidden_size = hidden_size
        self.lr_d = lr_d
        self.lambda_r1_ = lambda_r1_
        self.lambda_nonop = lambda_neg_
        self.reg_lambda_d = reg_lambda_d_
        self.name = name
        self.REWARD_THRESHOLD_D = REWARD_THRESHOLD_D
        self.optimizer_d = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr_d)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, ob_shape[0]])
        self.actions_onehot = tf.placeholder(dtype=tf.float32, shape=[None, ac_shape[0]])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None])
        self.dropout_rate_d = tf.placeholder(tf.float32)

        self.embedding_dim = 64
        self.nfm_state_embeddings = tf.Variable(tf.random.normal([self.ob_shape[0], self.embedding_dim], 0.0, 0.01), name=str(self.name) + '_fm_state_embeddings')  # usr_x_size * K
        self.nfm_action_embeddings = tf.Variable(tf.random.normal([self.ac_shape[0], self.embedding_dim], 0.0, 0.01), name=str(self.name) + '_fm_action_embeddings')  # usr_x_size * K
        self._build_network(self.name)

    def _build_network(self, name):
        with tf.variable_scope(str(name) + '_discriminator'):

            state_expanded_features = tf.keras.backend.repeat_elements(tf.expand_dims(self.state, 2), self.embedding_dim, axis=2)
            state_weighted_embeddings = tf.multiply(self.nfm_state_embeddings, state_expanded_features)  # [None x user_x_size x embedding_size]

            action_expanded_features = tf.keras.backend.repeat_elements(tf.expand_dims(self.actions_onehot, 2), self.embedding_dim, axis=2)
            action_weighted_embeddings_ = tf.multiply(self.nfm_action_embeddings, action_expanded_features)  # [None x user_x_size x embedding_size]
            action_weighted_embedding_ = tf.reduce_sum(action_weighted_embeddings_, axis=1)       # Since only 1 value is non zero (one-hot), takes the sum across the dimension to obtain the correct embedding

            # ________ FM __________
            FM_o1 = tf.reduce_sum(state_weighted_embeddings, axis=1)  # None * embedding_size
            # _________ sum_square _____________
            # get the summed up embeddings of features.
            summed_features_emb = tf.reduce_sum(state_weighted_embeddings, 1)  # None * K
            summed_features_emb_square = tf.square(summed_features_emb)  # None * K
            # _________ square_sum _____________
            squared_features_emb = tf.square(state_weighted_embeddings)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K
            FM_o2 = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
            fm_output = tf.reduce_sum([FM_o1, FM_o2], axis=0)
            self.d_out = tf.reduce_sum(tf.multiply(fm_output, action_weighted_embedding_), axis=1)

            self.loss = self.bce_loss(y_true=self.labels, y_pred=self.d_out) #+ self.reg_lambda_d * l2_reg
            self.mean_abs_error = self.mae(y_true=self.labels, y_pred=self.d_out)
            self.train_op = self.optimizer_d.minimize(loss=self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name))

    def create_embedding_layers(self, input_tensor, num_features, embedding_dimension):
        embeddings = []
        for i in range(num_features):
            # Slice the i-th feature column [batch_size, 1]
            feature_i = tf.slice(input_tensor, [0, i], [-1, 1])
            embedding = tf.keras.layers.Embedding(input_dim=self.feature_max_val[i] + 1,
                                                  output_dim=embedding_dimension,
                                                  input_length=1)(feature_i)
            # Reshape from [batch_size, 1, embedding_dimension] to [batch_size, embedding_dimension]
            embedding = tf.reshape(embedding, [-1, embedding_dimension])
            embeddings.append(embedding)
        return embeddings

    def get_feedback(self, state, actions_onehot, dropout_):
        """
        Get feedback from the adversarial discriminator based on observation-action pairs.
        """
        feed_dict = {self.state: state, self.actions_onehot: actions_onehot, self.dropout_rate_d: dropout_}
        d_out = self.sess.run(self.d_out, feed_dict=feed_dict)
        return d_out

    def getloss(self, state, actions_onehot, labels_, dropout_):
        """
        Update the cooperative discriminator based on observation-action pairs and calculate loss.
        """
        feed_dict = {self.state: state, self.actions_onehot: actions_onehot, self.labels: labels_, self.dropout_rate_d: dropout_}
        d_loss, error = self.sess.run((self.loss, self.mean_abs_error), feed_dict=feed_dict)
        return d_loss, error

    def update_n_getloss(self, state, actions_onehot, labels_, dropout_):
        """
        Update the cooperative discriminator based on observation-action pairs and calculate loss.
        """
        feed_dict = {self.state: state, self.actions_onehot: actions_onehot, self.labels: labels_, self.dropout_rate_d: dropout_}
        test_out = self.sess.run(self.d_out, feed_dict=feed_dict)
        _, d_loss, error = self.sess.run((self.train_op, self.loss, self.mean_abs_error), feed_dict=feed_dict)
        return d_loss, error

def one_hot(pt_acs_all_batch):
    return np.array([np.eye(ac_size)[x] for x in pt_acs_all_batch])


def train(no_epochs, replace_target_freq, phase='train'):
    # print("%Epoch\tP_Loss\tP_Error\tDA_Loss\tDA_Error\tDC_Loss\tDC_Error\tMeanQ_Survvd\tMeanQ_Disesd")
    print("%Epoch\tP_Error\tDA_Error\tDC_Error\tD_A_opt_loss_epoch\tD_A_gen_loss_epoch\tD_C_hq_loss_epoch\tD_C_no_loss_epoch\tMeanQ_Survvd\tMeanQ_Disesd\tmean_D_C_op_error\tmean_D_C_no_error")

    for epoch in range(1, no_epochs):
        policy_loss_epoch = []
        policy_error_epoch = []
        D_A_loss_epoch = []
        D_A_error_epoch = []
        D_C_loss_epoch = []
        D_C_error_epoch = []
        z_D_C_error_parts_op = []
        z_D_C_error_parts_no = []

        D_A_gen_loss_epoch = []
        D_A_opt_loss_epoch = []
        D_C_hq_loss_epoch = []
        D_C_no_loss_epoch = []

        D_A_feedback_epoch = []
        D_C_feedback_epoch = []
        for _ in range(int(p_total_train_steps / args.batch_size)):

            # 1. Extract optimal and non-optimal instances from exp and train_policy. (Note that all train instances were divided to exp and train_policy in data processing.)
            batch_indices_exp_optimal = np.random.choice(exp_ob_ac_optml.shape[0], args.batch_size, replace=False)
            batch_indices_exp_nonop_sampled = np.random.choice(exp_ob_ac_nonop_sampled.shape[0], args.batch_size, replace=False)
            # Takes batch_size * 2 to match the real sample batch size
            batch_indices_train_policy_all = np.random.choice(train_policy_obs_all.shape[0], args.batch_size, replace=False)

            exp_optml_ob_ac_batch = exp_ob_ac_optml[batch_indices_exp_optimal]

            exp_optml_ob_batch = exp_obs_optml[batch_indices_exp_optimal]
            exp_optml_ac_batch = exp_acs_optml[batch_indices_exp_optimal]
            exp_nonop_ob_batch_sampled = exp_obs_nonop_sampled[batch_indices_exp_nonop_sampled]
            exp_nonop_ac_batch_sampled = exp_acs_nonop_sampled[batch_indices_exp_nonop_sampled]

            train_policy_obs_all_batch = train_policy_obs_all[batch_indices_train_policy_all]
            train_policy_acs_all_batch = train_policy_acs_all[batch_indices_train_policy_all]
            train_policy_rewards_all_batch = train_policy_mortality_reward_all[batch_indices_train_policy_all]

            # 2. Obtain actions for train policy observations.
            train_pt_acs_all_batch = sess.run(policy.mainQN.predicted_best_action, feed_dict={policy.mainQN.state: train_policy_obs_all_batch, policy.mainQN.dropout_rate: args.dropout})
            train_pt_acs_onehot_all_batch = one_hot(train_pt_acs_all_batch)
            train_pt_ob_ac_all_batch = np.concatenate([train_policy_obs_all_batch, train_pt_acs_onehot_all_batch], axis=1)

            hq_ob_optmal_batch = np.concatenate([exp_optml_ob_batch, train_policy_obs_all_batch], axis=0)
            hq_ac_optmal_batch = np.concatenate([exp_optml_ac_batch, train_pt_acs_onehot_all_batch], axis=0)

            X_L_opt_A = exp_optml_ob_ac_batch
            X_L_gen_A = train_pt_ob_ac_all_batch

            X_L_opt_A_state = exp_optml_ob_batch
            X_L_gen_A_state = train_policy_obs_all_batch
            X_L_hq_C_state = hq_ob_optmal_batch
            X_L_hq_C_acs = hq_ac_optmal_batch
            X_L_no_C_state = exp_nonop_ob_batch_sampled
            X_L_no_C_acs = exp_nonop_ac_batch_sampled

            Y_L_opt_A = np.ones(len(X_L_opt_A_state))
            Y_L_gen_A = np.zeros(len(X_L_gen_A_state))
            Y_L_hq_C = np.ones(len(X_L_hq_C_state))
            Y_L_no_C = np.zeros(len(X_L_no_C_state))

            # 2. Prepare x and y for training:
            #   X: ob_ac concatenations for exp_op, exp_no and train_policy. Note that actions for train policy observations are taken from the policy learnt.
            #   Y: 1 for optimal instances (exp_op), 0 for rest (exp_no and train_all).
            train_policy_next_obs_all_batch = train_policy_next_obs_all[batch_indices_train_policy_all]
            train_policy_notdone_all_batch = train_policy_p_notdone_all[batch_indices_train_policy_all]
            train_ob_ac_all_batch = np.concatenate([train_policy_obs_all_batch, train_policy_acs_all_batch], axis=1)
            # Update the cooperative discriminator (D_C)
            X_L_hq_C_state_shuffled_, X_L_hq_C_acs_shuffled_, Y_L_hq_C_shuffled_ = shuffle(X_L_hq_C_state, X_L_hq_C_acs, Y_L_hq_C, random_state=0)
            # halving the amount of samples since HQ composed of 2 (opt and gen) of batch sizes to avoid imbalance.
            X_L_hq_C_state_shuffled = X_L_hq_C_state_shuffled_[: len(Y_L_no_C)]
            X_L_hq_C_acs_shuffled = X_L_hq_C_acs_shuffled_[: len(Y_L_no_C)]
            Y_L_hq_C_shuffled = Y_L_hq_C_shuffled_[: len(Y_L_no_C)]

            X_L_no_C_state_shuffled, X_L_no_C_acs_shuffled, Y_L_no_C_shuffled = shuffle(X_L_no_C_state, X_L_no_C_acs, Y_L_no_C, random_state=0)
            Dc_loss_batch_2, Dc_error_batch_2 = discriminator_Cop.update_n_getloss(X_L_no_C_state_shuffled, X_L_no_C_acs_shuffled, Y_L_no_C_shuffled, args.dropout_d)

            Y_L_opt_C = np.ones(len(exp_optml_ob_batch))
            Dc_loss_batch_0, Dc_error_batch_0 = discriminator_Cop.update_n_getloss(exp_optml_ob_batch, exp_optml_ac_batch, Y_L_opt_C, args.dropout_d)

            z_D_C_error_parts_op.append(Dc_error_batch_0)
            z_D_C_error_parts_no.append(Dc_error_batch_2)
            D_C_no_loss_epoch.append(Dc_error_batch_2)

            Dc_loss_batch_1, Dc_error_batch_1 = discriminator_Cop.update_n_getloss(X_L_hq_C_state_shuffled, X_L_hq_C_acs_shuffled, Y_L_hq_C_shuffled, args.dropout_d)
            D_C_loss_epoch.append(np.mean([Dc_loss_batch_1, Dc_loss_batch_2]))
            D_C_error_epoch.append(np.mean([Dc_error_batch_1, Dc_error_batch_2]))
            D_C_hq_loss_epoch.append(Dc_loss_batch_1)
            Dc_feedback = discriminator_Cop.get_feedback(train_policy_obs_all_batch, train_policy_acs_all_batch, args.dropout)

            # Update the adversarial discriminator (D_A)
            X_L_opt_A_shuffled, Y_L_opt_A_shuffled = shuffle(X_L_opt_A, Y_L_opt_A, random_state=0)
            X_L_gen_A_shuffled, Y_L_gen_A_shuffled = shuffle(X_L_gen_A, Y_L_gen_A, random_state=0)
            Da_loss_batch_1, Da_error_batch_1 = discriminator_Adv.update_n_getloss(X_L_opt_A_shuffled, Y_L_opt_A_shuffled, args.dropout_d)
            Da_loss_batch_2, Da_error_batch_2 = discriminator_Adv.update_n_getloss(X_L_gen_A_shuffled, Y_L_gen_A_shuffled, args.dropout_d)

            # Update the Policy (Pi_theta)
            # Get feedback from both discriminators
            Da_feedback = discriminator_Adv.get_feedback(train_ob_ac_all_batch, args.dropout)

            policy_loss_batch, policy_error_batch = policy.update_mainQ_and_get_loss_with_DaDc_feedback(train_policy_obs_all_batch, train_pt_acs_onehot_all_batch, train_policy_next_obs_all_batch, train_policy_notdone_all_batch, args.dropout, train_policy_rewards_all_batch, Da_feedback, Dc_feedback)

            if epoch % replace_target_freq == 0:
                policy.update_target()

            policy_loss_epoch.append(policy_loss_batch)
            policy_error_epoch.append(policy_error_batch)
            D_A_loss_epoch.append(np.mean([Da_loss_batch_1, Da_loss_batch_2]))
            D_A_error_epoch.append(np.mean([Da_error_batch_1, Da_error_batch_2]))
            D_A_opt_loss_epoch.append(Da_loss_batch_1)
            D_A_gen_loss_epoch.append(Da_loss_batch_2)

            D_A_feedback_epoch.extend(Da_feedback)
            D_C_feedback_epoch.extend(Dc_feedback)

        # 5. PLOTTING results:
        mean_policy_error = np.mean(policy_error_epoch)
        mean_D_A_error = np.mean(D_A_error_epoch)
        mean_D_C_error = np.mean(D_C_error_epoch)
        mean_D_C_op_error = np.mean(z_D_C_error_parts_op)
        mean_D_C_no_error = np.mean(z_D_C_error_parts_no)

        train_q_values_survived = sess.run(policy.mainQN.Q, feed_dict={policy.mainQN.state: train_policy_obs_survived, policy.mainQN.actions_onehot: train_policy_acs_survived, policy.mainQN.dropout_rate: 0.0})
        train_q_values_deceased = sess.run(policy.mainQN.Q, feed_dict={policy.mainQN.state: train_policy_obs_deceased, policy.mainQN.actions_onehot: train_policy_acs_deceased, policy.mainQN.dropout_rate: 0.0})

        print("%s E %d/%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f"
              % (phase, epoch, no_epochs, mean_policy_error, mean_D_A_error, mean_D_C_error, np.mean(D_A_opt_loss_epoch), np.mean(D_A_gen_loss_epoch), np.mean(D_C_hq_loss_epoch), np.mean(D_C_no_loss_epoch), np.mean(train_q_values_survived), np.mean(train_q_values_deceased), mean_D_C_op_error, mean_D_C_no_error))

    # generating actions
    data_file = '../Data/<data_file>.pkl'   # processed data file.
    D = utils.load_dict(data_file)
    D = copy.deepcopy(D)
    exp_data = D['exp_data']
    exp_data['type'] = 'EXP'

    data_nonop = D['train_policy_df_nonop']
    data_nonop['type'] = 'NOP'
    data_op = D['train_policy_df_optml']
    data_op['type'] = 'OPT'

    state_cols = D['UPDATED_STATE_COLs']
    action_cols = D['ACTION_COLs']
    merged_df = pd.concat([exp_data, data_nonop, data_op])

    # A simple check to see if steps are optimal and nonoptimal intermittently for the same patient. Which is a main hypothesis of our approach
    result = TrajUtils.check_for_intermittent_opt_nonopt_steps(merged_df)
    print(f'\nDoes training data have opt and nonopt steps intermittently for given patients? ', result)

    df_altered = TrajUtils.create_trajectories(sess, 'SI', merged_df, policy, state_cols, action_cols)
    df_altered.to_csv('./SI_trajectories/SI_ALL_Trajectories.csv', index=False)

    print('................................... Done, trajectories are saved.  ...................................')


if __name__ == '__main__':

    # Hyperparameters
    folder_outputs = "../outputs/"
    dict_filename = '../Data/<data_file>.pkl'   # processed data file.

    output_graphs = os.path.basename(__file__)
    filename_postfix = '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_graph_output = folder_outputs + "graphs_" + output_graphs.split(".")[0] + filename_postfix
    file_args_output = folder_outputs + "args_" + output_graphs.split(".")[0] + filename_postfix
    folder_outputs_maingraphs = folder_outputs + 'main_graphs/'
    train_file_prefix_maingraphs = folder_outputs_maingraphs + "train_graphs_" + output_graphs.split(".")[0] + filename_postfix + '_'
    test_file_prefix_maingraphs = folder_outputs_maingraphs + "test_graphs_" + output_graphs.split(".")[0] + filename_postfix + '_'

    output_all_plots = "./result/train_graphs_" + output_graphs.split(".")[0] + filename_postfix + '_' + utils.TRAIN_PHASE1 + '_output_all.csv'
    args = parse_args()
    tf.compat.v1.set_random_seed(2021)
    dropout_test = 0.0
    dropout_d = args.dropout_d

    policy_loss_list = []
    policy_error_list = []
    D_A_feedback_list = []
    D_C_feedback_list = []
    DA_loss_list = []
    DC_loss_list = []
    DA_error_list = []
    DC_error_list = []

    pretrain_q_values_survived_histo = []
    pretrain_q_values_diseased_histo = []
    pretrain_nonop_q_values_histo = []
    pretrain_optml_q_values_histo = []

    train_mean_q_values_survived = []
    train_mean_q_values_diseased = []
    train_nonop_q_values = []
    train_optml_q_values = []
    train_q_values_survived_histo = []
    train_q_values_diseased_histo = []
    train_nonop_q_values_histo = []
    train_optml_q_values_histo = []

    test_mean_q_values_survived = []
    test_mean_q_values_diseased = []
    test_nonop_q_values = []
    test_optml_q_values = []
    test_q_values_survived_histo = []
    test_q_values_diseased_histo = []
    test_nonop_q_values_histo = []
    test_optml_q_values_histo = []

    dict = utils.load_dict(dict_filename)
    UPDATED_STATE_COLs = dict['UPDATED_STATE_COLs']
    ACTION_COLs = dict['ACTION_COLs']
    UPDATED_NEXT_STATE_COLs = dict['UPDATED_NEXT_STATE_COLs']
    ORI_DATA_COLs_WO_TRAINTEST = dict['ORI_DATA_COLs_WO_TRAINTEST']

    exp_df_all = dict['exp_data']
    exp_df_survived = dict['exp_df_survived']
    exp_df_diseased = dict['exp_df_diseased']
    exp_df_optml = dict['exp_df_optml']
    exp_df_nonop = dict['exp_df_nonop']
    train_policy_df_all = dict['train_data_policy_']
    train_policy_df_survived = dict['train_policy_df_survived']
    train_policy_df_deceased = dict['train_policy_df_diseased']
    train_policy_df_optml = dict['train_policy_df_optml']
    train_policy_df_nonop = dict['train_policy_df_nonop']
    test_df_all = dict['test_data_']
    test_df_survived = dict['test_df_survived']
    test_df_diseased = dict['test_df_diseased']
    test_df_optml = dict['test_df_optml']
    test_df_nonop = dict['test_df_nonop']

    exp_obs_all = np.array(exp_df_all[UPDATED_STATE_COLs])
    exp_acs_all = np.array(exp_df_all[ACTION_COLs])
    exp_obs_optml = np.array(exp_df_optml[UPDATED_STATE_COLs])
    exp_acs_optml = np.array(exp_df_optml[ACTION_COLs])
    exp_obs_nonop = np.array(exp_df_nonop[UPDATED_STATE_COLs])
    exp_acs_nonop = np.array(exp_df_nonop[ACTION_COLs])

    ob_size = len(UPDATED_STATE_COLs)
    ac_size = len(ACTION_COLs)
    ob_shape = [ob_size]
    ac_shape = [ac_size]

    exp_ob_ac_all = np.concatenate([exp_obs_all, exp_acs_all], 1)
    exp_ob_ac_optml = np.concatenate([exp_obs_optml, exp_acs_optml], 1)
    exp_ob_ac_nonop = np.concatenate([exp_obs_nonop, exp_acs_nonop], 1)

    # Oversampling the low nonoptimal samples
    optml_labels = np.ones(len(exp_ob_ac_optml))
    nonop_labels = np.zeros(len(exp_ob_ac_nonop))
    X = np.concatenate((exp_ob_ac_optml, exp_ob_ac_nonop), axis=0)
    y = np.concatenate((optml_labels, nonop_labels), axis=0)

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    exp_ob_ac_nonop_sampled = X_resampled[y_resampled == 0]
    exp_obs_nonop_sampled, exp_acs_nonop_sampled = np.split(exp_ob_ac_nonop_sampled, [ob_size], axis=1)


    train_policy_obs_all = np.array(train_policy_df_all[UPDATED_STATE_COLs])
    train_policy_acs_all = np.array(train_policy_df_all[ACTION_COLs])
    train_policy_obs_optml = np.array(train_policy_df_optml[UPDATED_STATE_COLs])
    train_policy_acs_optml = np.array(train_policy_df_optml[ACTION_COLs])
    train_policy_obs_nonop = np.array(train_policy_df_nonop[UPDATED_STATE_COLs])
    train_policy_acs_nonop = np.array(train_policy_df_nonop[ACTION_COLs])

    train_policy_ob_ac_all = np.concatenate([train_policy_obs_all, train_policy_acs_all], 1)
    train_policy_ob_ac_optml = np.concatenate([train_policy_obs_optml, train_policy_acs_optml], 1)
    train_policy_ob_ac_nonop = np.concatenate([train_policy_obs_nonop, train_policy_acs_nonop], 1)

    train_policy_next_obs_all = np.array(train_policy_df_all[UPDATED_NEXT_STATE_COLs])
    train_policy_next_obs_optml = np.array(train_policy_df_optml[UPDATED_NEXT_STATE_COLs])
    train_policy_next_obs_all = np.nan_to_num(train_policy_next_obs_all)
    train_policy_next_obs_optml = np.nan_to_num(train_policy_next_obs_optml)

    train_policy_obs_survived = np.array(train_policy_df_survived[UPDATED_STATE_COLs])
    train_policy_acs_survived = np.array(train_policy_df_survived[ACTION_COLs])
    train_policy_obs_deceased = np.array(train_policy_df_deceased[UPDATED_STATE_COLs])
    train_policy_acs_deceased = np.array(train_policy_df_deceased[ACTION_COLs])

    test_obs_survived = np.array(test_df_survived[UPDATED_STATE_COLs])
    test_acs_survived = np.array(test_df_survived[ACTION_COLs])
    test_obs_diseased = np.array(test_df_diseased[UPDATED_STATE_COLs])
    test_acs_diseased = np.array(test_df_diseased[ACTION_COLs])
    test_obs_nonop = np.array(test_df_nonop[UPDATED_STATE_COLs])
    test_acs_nonop = np.array(test_df_nonop[ACTION_COLs])
    test_obs_optml = np.array(test_df_optml[UPDATED_STATE_COLs])
    test_acs_optml = np.array(test_df_optml[ACTION_COLs])

    train_policy_mortality_reward_all = np.array(train_policy_df_all[setting_4.REWARD_COL]) * train_policy_df_all['done']
    train_policy_mortality_reward_optml = np.array(train_policy_df_optml[setting_4.REWARD_COL]) * train_policy_df_optml['done']
    train_policy_mortality_reward_nonop = np.array(train_policy_df_nonop[setting_4.REWARD_COL]) * train_policy_df_nonop['done']
    train_policy_mortality_reward_survived = np.array(train_policy_df_survived[setting_4.REWARD_COL]) * train_policy_df_survived['done']
    train_policy_mortality_reward_diseased = np.array(train_policy_df_deceased[setting_4.REWARD_COL]) * train_policy_df_deceased['done']
    test_mortality_reward_all = np.array(test_df_all[setting_4.REWARD_COL]) * test_df_all['done']
    test_mortality_reward_survived = np.array(test_df_survived[setting_4.REWARD_COL]) * test_df_survived['done']
    test_mortality_reward_deceased = np.array(test_df_diseased[setting_4.REWARD_COL]) * test_df_diseased['done']

    ori_data_columns_wo_traintest = ORI_DATA_COLs_WO_TRAINTEST
    beta = len(exp_df_nonop) / (len(exp_df_nonop) + len(exp_df_optml))
    p_total_train_steps = train_policy_df_all.shape[0]
    train_policy_p_notdone_all = 1-train_policy_df_all['done']
    train_policy_p_notdone_optml_all = 1 - train_policy_df_optml['done']
    print('beta: {}'.format(beta))
    Q_MAIN_SCOPE = "Q_Main"
    Q_TARGET_SCOPE = "Q_Target"

    with tf.Session() as sess:
        # Initialize PPO and Discriminator
        policy = Policy(sess, ob_size, ac_size, Q_MAIN_SCOPE, Q_TARGET_SCOPE, args.qnet_hidden_size, args.reward_Threshold_q, args.reg_lambda, args.lr, args.tau, args.gamma, args.D_rewards_weight)

        discriminator_Adv = Discriminator_Adv(sess, ob_shape, ac_shape, beta, args.discriminator_hidden_size, args.lr_dA, args.lambda_r1, args.lambda_neg, args.reg_lambda_d, 'D_A', args.reward_Threshold_d)
        discriminator_Cop = Discriminator_Cop(sess, ob_shape, ac_shape, beta, args.discriminator_hidden_size, args.lr_dC, args.lambda_r1, args.lambda_neg, args.reg_lambda_d, 'D_C', args.reward_Threshold_d)

        init = tf.global_variables_initializer()
        sess.run(init)
        start_time = time.time()

        # ----------------------- TRAIN -----------------------------------
        print("##############################################################################", output_graphs)
        # TRAIN PHASE 1

        # Save Args File
        with open(file_args_output + '.json', 'w') as file:
            json.dump(args.__dict__, file, indent=4)  # use `json.loads` to do the reverse
        # --------------------------------------------------------------------

        outputs_dict = {
            'policy_loss_list': policy_loss_list,
            'policy_error_list': policy_error_list,
            'DA_loss_list': DA_loss_list,
            'DC_loss_list': DC_loss_list,
            'DA_error_list': DA_error_list,
            'DC_error_list': DC_error_list,
            'pretrain_q_values_survived_histo': pretrain_q_values_survived_histo,
            'pretrain_q_values_diseased_histo': pretrain_q_values_diseased_histo,
            'pretrain_neg_q_values_histo': pretrain_nonop_q_values_histo,
            'pretrain_nonneg_q_values_histo': pretrain_optml_q_values_histo,
            'train_mean_q_values_survived': train_mean_q_values_survived,
            'train_mean_q_values_diseased': train_mean_q_values_diseased,
            'train_neg_q_values': train_nonop_q_values,
            'train_nonneg_q_values': train_optml_q_values,
            'train_q_values_survived_histo': train_q_values_survived_histo,
            'train_q_values_diseased_histo': train_q_values_diseased_histo,
            'train_neg_q_values_histo': train_nonop_q_values_histo,
            'train_nonneg_q_values_histo': train_optml_q_values_histo,
            'test_mean_q_values_survived': test_mean_q_values_survived,
            'test_mean_q_values_diseased': test_mean_q_values_diseased,
            'test_neg_q_values': test_nonop_q_values,
            'test_nonneg_q_values': test_optml_q_values,
            'test_q_values_survived_histo': test_q_values_survived_histo,
            'test_q_values_diseased_histo': test_q_values_diseased_histo,
            'test_neg_q_values_histo': test_nonop_q_values_histo,
            'test_nonneg_q_values_histo': test_optml_q_values_histo,
            'EPOCHS_PER_PLOT_DATA': args.EPOCHS_PER_PLOT_DATA}

        np.save(file_graph_output, outputs_dict)

        end_time = time.time()
        print('Completed In: ' + str(timedelta(seconds=end_time - start_time)))