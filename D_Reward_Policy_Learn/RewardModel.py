import tensorflow as tf
import numpy as np


class Linear:
    def __init__(self, name, input_dims, output_dims, init_scale=0.1):
        self.w = tf.get_variable(name + "/w", [input_dims, output_dims], initializer=tf.random_normal_initializer(stddev=init_scale))
        self.b = tf.get_variable(name + "/b", [output_dims], initializer=tf.constant_initializer(0.0))

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b


def contrastive_loss_(penalty_scalar, good_reward, bad_reward):
    # Calculate squared Euclidean distance between the rewards
    squared_distance = tf.square(good_reward - bad_reward)
    # Calculate dynamic margin based on the maximum or average distance
    margin = tf.reduce_mean(squared_distance)
    # Calculate contrastive loss
    contrastive_loss = tf.maximum(0.0, margin - squared_distance)
    # Check if good rewards are greater than bad rewards
    rewards_check = tf.cast(tf.greater(good_reward, bad_reward), tf.float32)
    # Calculate the difference between good and bad rewards
    reward_diff = good_reward - bad_reward
    # Map the difference to a range between 0 and 1 using a sigmoid function
    sigmoid_diff = tf.nn.sigmoid(reward_diff)
    # Scale the sigmoid difference by the reward threshold
    adaptive_penalty = penalty_scalar * sigmoid_diff
    # Assign a high loss penalty if good rewards are not greater than bad rewards
    contrastive_loss = contrastive_loss * rewards_check + (2.0 - rewards_check) * adaptive_penalty

    # Compute the mean loss
    mean_loss = tf.reduce_mean(contrastive_loss)

    return mean_loss


def contrastive_loss(good_reward, bad_reward, reward_distance_type):
    distance = None
    if reward_distance_type == 'Manhattan':
        print('Distance Metric used : Manhattan')
        distance = tf.reduce_sum(tf.abs(good_reward - bad_reward), axis=-1)  # Calculate Manhattan distance between the rewards
    elif reward_distance_type == 'Chebyshev':
        print('Distance Metric used : Chebyshev')
        distance = tf.reduce_max(tf.abs(good_reward - bad_reward), axis=-1)  # Calculate Chebyshev distance between the rewards
    elif reward_distance_type == 'Cosine':
        print('Distance Metric used : Cosine')
        distance = 1.0 - tf.reduce_sum(tf.multiply(good_reward, bad_reward), axis=-1) / (tf.norm(good_reward, axis=-1) * tf.norm(bad_reward, axis=-1))  # Calculate Cosine distance between the rewards
    else:  # Default distance is Euclidean distance between the rewards
        print('Distance Metric used : Euclidean')
        distance = tf.square(good_reward - bad_reward)
    margin = tf.reduce_mean(distance)  # Calculate dynamic margin based on the maximum or average distance
    contrastive_loss = tf.maximum(0.0, margin - distance)  # Calculate contrastive loss
    mean_loss = tf.reduce_mean(contrastive_loss)  # Compute the mean loss
    return mean_loss


class RewardNet():
    def __init__(self, include_action, ob_dim, ac_dim, REWARD_THRESHOLD, middle_layers, dropout_rates, activations, scale_factor=5, reward_smooth_func='sigmoid'):
        in_dims = ob_dim + ac_dim if include_action else ob_dim

        with tf.variable_scope('weights') as param_scope:
            fcs = []
            last_dims = in_dims
            l = 0
            for layer_size in middle_layers:
                fcs.append(Linear('fc%d' % (l + 1), last_dims, layer_size))  # (l+1) is gross, but for backward compatibility
                last_dims = layer_size
                l = l + 1
            fcs.append(Linear('fc%d' % (l + 1), last_dims, 1))

        self.fcs = fcs
        self.in_dims = in_dims
        self.include_action = include_action
        self.param_scope = param_scope
        self.REWARD_THRESHOLD = REWARD_THRESHOLD
        self.dropout_rates = dropout_rates
        self.activations = activations
        self.scale_factor = scale_factor
        self.reward_smooth_func = reward_smooth_func

    def input_preprocess(self, obs_g, acs_g, obs_b, acs_b):
        assert len(obs_g) == len(acs_g)
        assert len(obs_b) == len(acs_b)

        return \
            np.concatenate((obs_g, acs_g), axis=1) if self.include_action \
                else obs_g, np.concatenate((obs_b, acs_b), axis=1) if self.include_action \
                else obs_b

    def build_input_placeholder(self, name):
        return tf.placeholder(tf.float32, [None, self.in_dims], name=name)

    def build_reward_smooth_clipping(self, x):
        i = 0
        for fc in self.fcs[:-1]:
            x = self.activations[i](fc(x))
            x = tf.nn.dropout(x, rate=self.dropout_rates[i])
            i = i + 1
        r = tf.squeeze(self.fcs[-1](x), axis=1)

        smooth_rewards = None
        if self.reward_smooth_func == 'sigmoid':
            print('----------- Using Sigmoid scaling for rewards -----------')
            # Replace abrupt clipping with smooth clipping
            sigmoid_r = tf.math.sigmoid(r * self.scale_factor)
            smooth_rewards = 2 * self.REWARD_THRESHOLD * sigmoid_r - self.REWARD_THRESHOLD
        elif self.reward_smooth_func == 'tanh':
            print('----------- Using tanh scaling for rewards -----------')
            # Scaled by tanh function
            tanh_scaled_r = tf.math.tanh(r)
            # Scale the rewards to be in range of -REWARD_THRESHOLD and +REWARD_THRESHOLD
            smooth_rewards = tanh_scaled_r * self.REWARD_THRESHOLD

        return x, smooth_rewards

    def build_weight_decay(self):
        weight_decay = 0.
        for fc in self.fcs:
            weight_decay += tf.reduce_sum(fc.w ** 2)
        return weight_decay


class Model(object):
    def __init__(self, net: RewardNet, batch_size=64, lr=1e-3, reward_distance_type='Euclidean', ce_weight=1.0):
        self.B = batch_size
        self.lr = lr
        self.net = net
        self.reward_distance_type = reward_distance_type
        self.ce_weight = ce_weight

        self.good_steps = net.build_input_placeholder('x')  # tensor shape of [B*steps_x] + input_dims
        self.good_traj_lengths = tf.placeholder(tf.int32, [self.B])  # B-lengthed vector indicating the size of each steps_x
        self.bad_steps = net.build_input_placeholder('y')  # tensor shape of [B*steps_y] + input_dims
        self.bad_traj_lengths = tf.placeholder(tf.int32, [self.B])  # B-lengthed vector indicating the size of each steps_y
        self.l = tf.placeholder(tf.int32, [self.B])  # [0 when x is better 1 when y is better]

        self.l2_reg = tf.placeholder(tf.float32, [])

        # Graph ops for training
        _, self.rs_good_steps = net.build_reward_smooth_clipping(self.good_steps)  # rewards for all steps
        self.v_good_trajs_means = tf.stack([tf.reduce_mean(rs_x) for rs_x in tf.split(self.rs_good_steps, self.good_traj_lengths, axis=0)])

        _, self.rs_bad_steps = net.build_reward_smooth_clipping(self.bad_steps)  # rewards for all steps
        self.v_bad_trajs_means = tf.stack([tf.reduce_mean(rs_y) for rs_y in tf.split(self.rs_bad_steps, self.bad_traj_lengths, axis=0)])
        penalty_scalar = self.net.REWARD_THRESHOLD * self.net.scale_factor
        self.pairwise_loss = contrastive_loss(self.v_good_trajs_means, self.v_bad_trajs_means, self.reward_distance_type)

        self.l_updated = tf.concat([self.l, tf.ones_like(self.l)], axis=0)  # Add a tensor of ones too so that both (good, bad) and (bad, good) pairs are supported
        self.logits = tf.concat([tf.stack([self.v_good_trajs_means, self.v_bad_trajs_means], axis=1), tf.stack([self.v_bad_trajs_means, self.v_good_trajs_means], axis=1)], axis=0)

        # Stack the tensors
        indices = tf.range(0, tf.shape(self.l_updated)[0])
        # Shuffle the indices
        shuffled_indices = tf.random.shuffle(indices)

        # Use the shuffled indices to select elements from the tensors
        l_updated = tf.gather(self.l_updated, shuffled_indices)
        logits = tf.gather(self.logits, shuffled_indices)

        self.ce_loss_temp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=l_updated)
        self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=l_updated))

        # Regularizer Ops
        weight_decay = net.build_weight_decay()
        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.v_bad_trajs_means, self.v_good_trajs_means), tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.zeros_like(pred)), tf.float32))

        self.optim = tf.train.AdamOptimizer(self.lr)
        self.loss = self.pairwise_loss + self.ce_weight * self.ce_loss
        gradients, variables = zip(*self.optim.compute_gradients(self.loss + self.l2_loss, var_list=self.parameters(train=True)))
        gradients, _ = tf.clip_by_global_norm(gradients, 100.0)  # Clip gradients with a threshold of 5.0 (or any suitable value)
        self.update_op = self.optim.apply_gradients(zip(gradients, variables))
        self.saver = tf.train.Saver(var_list=self.parameters(train=False), max_to_keep=0)

    def parameters(self, train=False):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.net.param_scope.name)
        if not train:
            variables = [var for var in variables if not var.name.endswith('/W:0')]
        return variables


def print_params(sess):
    params = sess.run(tf.trainable_variables())
    for param, value in zip(tf.trainable_variables(), params):
        print(f"{param.name}: {np.mean(value)}")


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def prepare_data(df_trajectories, state_cols, action_cols):
    # groupby fruit and color columns
    grouped_df = df_trajectories.groupby(['stay_id'])

    # define a function to apply to each group
    def get_tuples(group):
        return (group[state_cols].values, group[action_cols].values)

    # apply the function to each group and convert the result to a list
    return grouped_df.apply(get_tuples).tolist()
