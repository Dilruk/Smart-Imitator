import argparse
import tensorflow as tf
import Utils.utils_main as utils
import pickle
import os
from datetime import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description="D2_Policy_Learning.")
    parser.add_argument('--model_timestamp', type=str, default="20240222_141901", help='Timestamp of the reward model to be loaded.')
    parser.add_argument('--policy_epochs', nargs='?', default=11, help='Epochs to train the policy.')
    parser.add_argument('--policy_epochs_per_plot', nargs='?', default=2, help='Epochs per plotting.')
    parser.add_argument('--policy_lr', type=float, default=0.001, help='Contribution from CE Loss.')
    parser.add_argument('--policy_hidden_size', nargs='?', default=128, help='Contribution from CE Loss.')
    parser.add_argument('--policy_embed_size', nargs='?', default=64, help='Contribution from CE Loss.')
    parser.add_argument('--policy_update_target_freq', type=int, nargs='?', default=10, help='Contribution from CE Loss.')
    parser.add_argument('--policy_gamma', nargs='?', default=0.99, help='Contribution from CE Loss.')
    parser.add_argument('--policy_batch_size', nargs='?', default=1024, help='Contribution from CE Loss.')
    parser.add_argument('--buffer_save_filepath', type=str, default="./save_data_buffer/data_buffer_wth_generated_rewards", help='Filepath to save the data buffer used to store the .')
    return parser.parse_args()

def load_args_from_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

class QNetwork:
    def __init__(self, state_dim, action_dim, hidden_size, embed_size, scope_name):
        with tf.compat.v1.variable_scope(scope_name):  # Use the provided scope_name here
            self.scope = scope_name  # Save the scope_name as an attribute

            self.state_placeholder = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name="qnet_state")

            # ----------------- INTERACTION PART -----------------------
            self.state_embeddings = tf.Variable(tf.random.normal([state_dim, embed_size], stddev=tf.sqrt(2. / state_dim)), trainable=True)
            self.action_embeddings = tf.Variable(tf.random.normal([action_dim, embed_size], stddev=tf.sqrt(2. / action_dim)), trainable=True)

            state_input_expanded = tf.expand_dims(self.state_placeholder, axis=-1)                                                              # [None, state_size, 1]
            state_embeddings_matrix = tf.tile(tf.expand_dims(self.state_embeddings, axis=0), [tf.shape(self.state_placeholder)[0], 1, 1])            # [None, state_size, emb_size]
            state_embedding_scaled = tf.multiply(state_input_expanded, state_embeddings_matrix)                                                 # [None, state_size, emb_size]
            state_embeddings_sum = tf.reduce_sum(state_embedding_scaled, axis=1, keepdims=True)                                                 # [None, 1, emb_size]
            action_embeddings_matrix = tf.tile(tf.expand_dims(self.action_embeddings, axis=0), [tf.shape(self.state_placeholder)[0], 1, 1])          # [None, action_size, emb_size]
            action_values_interactions_1 = tf.squeeze(tf.matmul(state_embeddings_sum, action_embeddings_matrix, transpose_b=True), axis=1)      # [None, action_size]

            # ----------------- DNN PART 1: Through state embeddings -----------------------
            state_input = tf.squeeze(state_embeddings_sum, axis=1)
            layer1 = tf.layers.dense(state_input, hidden_size, activation=tf.nn.relu)
            layer2 = tf.layers.dense(layer1, int(2 * hidden_size), activation=tf.nn.relu)
            layer3 = tf.layers.dense(layer2, hidden_size, activation=tf.nn.relu)
            layer4 = tf.layers.dense(layer3, int(hidden_size/2), activation=tf.nn.relu)
            action_values_dnn_2 = tf.layers.dense(layer4, action_dim, activation=None)

            #  --------------- DNN PART 2: Through states (no embeddings) ---------------------
            layer21 = tf.layers.dense(self.state_placeholder, hidden_size, activation=tf.nn.relu)
            layer22 = tf.layers.dense(layer21, int(2 * hidden_size), activation=tf.nn.relu)
            layer23 = tf.layers.dense(layer22, hidden_size, activation=tf.nn.relu)
            layer24 = tf.layers.dense(layer23, int(hidden_size/2), activation=tf.nn.relu)
            # Output action values for each action
            action_values_dnn_3 = tf.layers.dense(layer24, action_dim, activation=None)

            # Determine the best action
            self.action_values = action_values_interactions_1 + action_values_dnn_2 + action_values_dnn_3
            self.best_action = tf.argmax(self.action_values, axis=1)



class Agent:
    def __init__(self, state_dim, action_dim, hidden_size, embed_size, state_cols, action_cols, next_state_cols, lr=1e-3, update_target_every=5, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_cols = state_cols
        self.action_cols = action_cols
        self.next_state_cols = next_state_cols
        self.q_network = QNetwork(state_dim, action_dim, hidden_size, embed_size, "q_network")
        self.target_network = QNetwork(state_dim, action_dim, hidden_size, embed_size, "target_network")
        self.lr = lr
        self.update_target_every = update_target_every
        self.gamma = gamma

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.action_placeholder = tf.compat.v1.placeholder(tf.float32, [None, action_dim], name="action")
        self.reward_placeholder = tf.compat.v1.placeholder(tf.float32, [None], name="reward")
        self.next_state_placeholder = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name="next_state")

        curr_q_values = self.q_network.action_values
        curr_q_value = tf.reduce_sum(self.action_placeholder * curr_q_values, axis=1)
        next_q_values = self.target_network.action_values  # changed to target network

        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        self.target = self.reward_placeholder + self.gamma * max_next_q_values
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights_list=[self.q_network.state_embeddings, self.q_network.action_embeddings])
        self.loss = tf.reduce_mean((curr_q_value - self.target) ** 2) + regularization_penalty
        # X When defining the training operation, modify it to include gradient clipping
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # Clip gradients; 5.0 is just an example value
        self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))


    def get_best_action(self, sess, state):
        return sess.run(self.q_network.best_action, feed_dict={self.q_network.state_placeholder: state})

    def update_target_network(self, sess):
        q_network_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.q_network.scope)
        target_network_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.target_network.scope)
        update_ops = []
        for q_variable, target_variable in zip(q_network_variables, target_network_variables):
            update_ops.append(target_variable.assign(q_variable.value()))
        sess.run(update_ops)

    def train(self, sess, epochs, batch_size, epochs_per_plot, train_file_prefix_maingraphs, clinical_reward_col, gen_reward_col, train_buffer, test_file_prefix_maingraphs, test_buffer):
        sess.run(tf.global_variables_initializer())
        epoch_range = range(epochs)
        # Convert DataFrames to NumPy arrays once, outside the training loop
        states_np = train_buffer[self.state_cols].values
        actions_np = train_buffer[self.action_cols].values
        rewards_np = train_buffer[clinical_reward_col].values + train_buffer[gen_reward_col].values
        next_states_np = train_buffer[self.next_state_cols].values

        # Use the tf.data API to handle batching and shuffling
        train_data = tf.data.Dataset.from_tensor_slices((states_np, actions_np, rewards_np, next_states_np))
        train_data = train_data.shuffle(buffer_size=len(train_buffer)).batch(batch_size).repeat(epochs)
        iterator = train_data.make_initializable_iterator()
        next_element = iterator.get_next()

        # Define outside of the loop to avoid creating multiple initializers
        iterator_init_op = iterator.initializer
        sess.run(iterator_init_op)
        try:
            # Training Phase
            for epoch in epoch_range:
                print(f'\rEpoch: {epoch + 1}/{epochs}')
                # Get the next batch of data
                states_batch, actions_batch, rewards_batch, next_states_batch = sess.run(next_element)

                feed_dict = {
                    self.q_network.state_placeholder: states_batch,
                    self.target_network.state_placeholder: next_states_batch,  # Add this line
                    self.action_placeholder: actions_batch,
                    self.reward_placeholder: rewards_batch,
                    self.next_state_placeholder: next_states_batch
                }
                loss, _ = sess.run((self.loss, self.train_op), feed_dict=feed_dict)

                # Reinitialize the iterator at the end of each epoch
                sess.run(iterator.initializer)

                if epoch % self.update_target_every == 0:
                    self.update_target_network(sess)

                if epoch % epochs_per_plot == 0:
                    q_vals = sess.run(self.q_network.action_values, feed_dict={self.q_network.state_placeholder: states_np})
                    utils.save_data_for_plots_v4(train_buffer, q_vals, train_file_prefix_maingraphs, gen_reward_col, epoch)
                    test_q_vals = sess.run(self.q_network.action_values, feed_dict={self.q_network.state_placeholder: test_buffer[self.state_cols].values})
                    utils.save_data_for_plots_v4(test_buffer, test_q_vals, test_file_prefix_maingraphs, gen_reward_col, epoch)

        except tf.errors.OutOfRangeError as e:
            print(f'Reached End of the dataset: {e}')
        # Testing Phase
        test_states, test_actions, test_rewards, test_next_states = test_buffer[self.state_cols], test_buffer[self.action_cols], test_buffer[gen_reward_col], test_buffer[self.next_state_cols]
        test_feed_dict = {
            self.q_network.state_placeholder: test_states.values,
            self.target_network.state_placeholder: test_next_states.values,  # Add this line
            self.action_placeholder: test_actions.values,
            self.reward_placeholder: test_rewards.values,
            self.next_state_placeholder: test_next_states.values
        }
        test_loss = sess.run(self.loss, feed_dict=test_feed_dict)
        print(f'Test Loss:\t{test_loss}')
        test_q_vals = sess.run(self.q_network.action_values, feed_dict={self.q_network.state_placeholder: test_buffer[self.state_cols].values})
        utils.save_data_for_plots_v4(test_buffer, test_q_vals, test_file_prefix_maingraphs, gen_reward_col, epochs)


def main():
    data_file = '../Data/<data_file>.pkl'   #processed data
    folder_outputs = "./outputs/"
    reward_model_save_dir = './save_reward_model/'

    D = utils.load_dict(data_file)
    state_cols = D['UPDATED_STATE_COLs']
    action_cols = D['ACTION_COLs']
    next_state_cols = D['UPDATED_NEXT_STATE_COLs']
    clinical_reward_col = 'reward'
    gen_reward_col = 'generated_reward'
    ob_dim_ = len(state_cols)
    ac_dim_ = len(action_cols)
    args = parse_args()

    epochs = int(args.policy_epochs)
    epochs_per_plot = int(args.policy_epochs_per_plot)
    update_target_freq = int(args.policy_update_target_freq)
    lr = args.policy_lr
    hidden_size = int(args.policy_hidden_size)
    embed_size = int(args.policy_embed_size)

    gamma = args.policy_gamma
    batch_size = int(args.policy_batch_size)
    buffer_save_filepath = str(args.buffer_save_filepath)
    timestamp = str(args.model_timestamp)

    save_dir = os.path.join(reward_model_save_dir, timestamp)
    loaded_args_json = load_args_from_config(save_dir + '/args_file.json')
    epoch_to_restore = int(loaded_args_json['Epochs'])

    output_graphs = os.path.basename(__file__)
    filename_postfix = '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder_outputs_maingraphs = folder_outputs + 'main_graphs/'
    train_file_prefix_maingraphs = folder_outputs_maingraphs + "train_graphs_" + output_graphs.split(".")[0] + filename_postfix + '_'
    test_file_prefix_maingraphs = folder_outputs_maingraphs + "test_graphs_" + output_graphs.split(".")[0] + filename_postfix + '_'

    #  ----------------------------------------------------------------------------------------------------------------------------
    #       1. Load the saved data buffer containing (state ,action, generated_reward, and next state) tuples
    #  ---------------------------------------------------------------------------------------------------------------------------------
    # Load the buffer
    train_buffer_save_filename = buffer_save_filepath + '_TRAIN_' + timestamp + '_' + str(epoch_to_restore) + '.pkl'
    test_buffer_save_filename = buffer_save_filepath + '_TEST_' + timestamp + '_' + str(epoch_to_restore) + '.pkl'

    # all_data_columns = copy.deepcopy(D['ACTION_COLs']) + copy.deepcopy(D['UPDATED_STATE_COLs'])
    with tf.compat.v1.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        with open(train_buffer_save_filename, 'rb') as train_f:
            train_buffer = pickle.load(train_f)
            with open(test_buffer_save_filename, 'rb') as test_f:
                # Use the test buffer
                test_buffer = pickle.load(test_f)
                agent = Agent(ob_dim_, ac_dim_, hidden_size, embed_size, state_cols, action_cols, next_state_cols, lr, update_target_freq, gamma)
                agent.train(sess, epochs, batch_size, epochs_per_plot, train_file_prefix_maingraphs, clinical_reward_col, gen_reward_col, train_buffer, test_file_prefix_maingraphs, test_buffer)

    print(f'.................. Completed ................. ')
    print(filename_postfix)




if __name__ == '__main__':
    main()
