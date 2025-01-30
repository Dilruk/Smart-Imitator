import copy
from RewardModel import *
from datetime import datetime
import os
import pandas as pd
import Utils.utils_main as utils
import random
import Utils.settings as settings
import argparse
import json
import numpy as np
import tensorflow as tf

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="D1_Reward_Learning.")
    parser.add_argument('--Epochs', type=int, nargs='?', default=20, help='Epochs.')
    parser.add_argument('--distance_type', type=str, nargs='?', default='Manhattan', help='Distance measure to measure the distance between good and bad rewards. Pick one between [Manhattan, Chebyshev, Cosine or Euclidean].')
    parser.add_argument('--reward_smooth_func', type=str, nargs='?', default='sigmoid', help='Reward smoothing function to be used, before scaling to MIN/MAX reward range. Pick one between [sigmoid and tanh]')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Contribution from CE Loss.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Contribution from CE Loss.')
    parser.add_argument('--l2_reg', nargs='?', default=0.001, help='Contribution from CE Loss.')
    parser.add_argument('--MINI_BATCH_SIZE', type=int, nargs='?', default=1024, help='Epochs.')
    parser.add_argument('--rewardnet_layers', metavar='N', type=int, nargs='+', default=[128, 64, 32, 8], help='Reward net layer sizes')
    parser.add_argument("--rewardnet_activations", nargs="+", type=str, default=["tanh", "relu", "leaky_relu", "relu"], help="Activation functions (choose from 'tanh', 'relu', or 'leaky_relu')")
    parser.add_argument('--layer_dropout_rates', metavar='F', type=float, nargs='+', default=[0.2, 0.1, 0.2, 0.2], help='Reward net dropout rates')
    parser.add_argument('--goodbad_pairs_mode', type=int, default=6, help='Takes integers 0-6 (inclusive)[].integraer defiing the level of pairs to be considered. higher levels introduce pairs with lesser distances, but allows more data points.')
    return parser.parse_args()

# Consider the method after this for full set of results
def compute_test_results_subset(model, Test_optml_data_, Test_nonop_data_, BATCH_SIZE):
    Test_optml_data = copy.deepcopy(Test_optml_data_)
    Test_nonop_data = copy.deepcopy(Test_nonop_data_)
    len_test_good_ = len(Test_optml_data)
    len_test_bad_ = len(Test_nonop_data)

    if BATCH_SIZE > len_test_good_:
        good_batch = copy.deepcopy(Test_optml_data)
        good_batch_fill_batch = random.choices(Test_optml_data, k=(BATCH_SIZE - len_test_good_))
        good_batch.extend(good_batch_fill_batch)
    else:
        good_batch = random.choices(Test_optml_data, k=BATCH_SIZE)

    if BATCH_SIZE > len_test_bad_:
        bad_batch = copy.deepcopy(Test_nonop_data)
        bad_batch_fill_batch = random.choices(Test_nonop_data, k=(BATCH_SIZE - len_test_bad_))
        bad_batch.extend(bad_batch_fill_batch)
    else:
        bad_batch = random.choices(Test_nonop_data, k=BATCH_SIZE)

    x_test = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in good_batch], axis=0)
    y_test = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in bad_batch], axis=0)
    x_split_test = np.array([len(ob[0]) for ob in good_batch])
    y_split_test = np.array([len(ob[0]) for ob in bad_batch])

    feed_dict_test = {
        model.good_steps: x_test,
        model.bad_steps: y_test,
        model.good_traj_lengths: x_split_test,
        model.bad_traj_lengths: y_split_test,
        model.l: [0] * BATCH_SIZE,
        model.l2_reg: 0.0}

    pairwise_loss_test, acc_test, v_good_trajs_means_test, v_bad_trajs_means_test = sess.run([model.pairwise_loss, model.acc, model.v_good_trajs_means, model.v_bad_trajs_means], feed_dict=feed_dict_test)

    mean_pairwise_loss_test_ = np.mean(pairwise_loss_test)
    mean_acc_test_ = np.mean(acc_test)

    mean_r_per_good_traj_test_ = np.mean(v_good_trajs_means_test)
    mean_r_per_bad_traj_test_ = np.mean(v_bad_trajs_means_test)
    return mean_pairwise_loss_test_, mean_acc_test_, mean_r_per_good_traj_test_, mean_r_per_bad_traj_test_


def compute_test_results(model, Test_optml_data_, Test_nonop_data_, BATCH_SIZE):
    Test_optml_data = copy.deepcopy(Test_optml_data_)
    Test_nonop_data = copy.deepcopy(Test_nonop_data_)
    len_test_good_ = len(Test_optml_data)
    len_test_bad_ = len(Test_nonop_data)

    if BATCH_SIZE > len_test_good_:
        good_batch = copy.deepcopy(Test_optml_data)
        good_batch_fill_batch = random.choices(Test_optml_data, k=(BATCH_SIZE - len_test_good_))
        good_batch.extend(good_batch_fill_batch)
    else:
        good_batch = random.choices(Test_optml_data, k=BATCH_SIZE)

    if BATCH_SIZE > len_test_bad_:
        bad_batch = copy.deepcopy(Test_nonop_data)
        bad_batch_fill_batch = random.choices(Test_nonop_data, k=(BATCH_SIZE - len_test_bad_))
        bad_batch.extend(bad_batch_fill_batch)
    else:
        bad_batch = random.choices(Test_nonop_data, k=BATCH_SIZE)

    x_test = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in good_batch], axis=0)
    y_test = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in bad_batch], axis=0)
    x_split_test = np.array([len(ob[0]) for ob in good_batch])
    y_split_test = np.array([len(ob[0]) for ob in bad_batch])

    feed_dict_test = {
        model.good_steps: x_test,
        model.bad_steps: y_test,
        model.good_traj_lengths: x_split_test,
        model.bad_traj_lengths: y_split_test,
        model.l: [0] * BATCH_SIZE,
        model.l2_reg: 0.0}

    pairwise_loss_test, acc_test, v_good_trajs_means_test, v_bad_trajs_means_test = sess.run([model.pairwise_loss, model.acc, model.v_good_trajs_means, model.v_bad_trajs_means], feed_dict=feed_dict_test)

    mean_pairwise_loss_test_ = np.mean(pairwise_loss_test)
    mean_acc_test_ = np.mean(acc_test)
    median_pairwise_loss_test_ = np.median(pairwise_loss_test)
    median_acc_test_ = np.median(acc_test)

    mean_r_per_good_traj_test_ = np.mean(v_good_trajs_means_test)
    mean_r_per_bad_traj_test_ = np.mean(v_bad_trajs_means_test)
    median_r_per_good_traj_test_ = np.median(v_good_trajs_means_test)
    median_r_per_bad_traj_test_ = np.median(v_bad_trajs_means_test)
    return mean_pairwise_loss_test_, mean_acc_test_, median_pairwise_loss_test_, median_acc_test_, mean_r_per_good_traj_test_, mean_r_per_bad_traj_test_, median_r_per_good_traj_test_, median_r_per_bad_traj_test_


def save_args_file(args, config_file):
    # save args to a file.
    with open(config_file, "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('TIME_STAMP: ' + timestamp)
    data_file = '../Data/<data_file>.pkl'   # processed data file
    reward_model_save_dir = './save_reward_model/'
    save_dir = os.path.join(reward_model_save_dir, timestamp)
    D = utils.load_dict(data_file)
    state_cols = D['UPDATED_STATE_COLs']
    action_cols = D['ACTION_COLs']
    ob_dim_ = len(state_cols)
    ac_dim_ = len(action_cols)

    args = parse_args()
    scale_factor = args.scale_factor
    lr = float(args.lr)
    ce_weight = args.ce_weight
    Epochs = args.Epochs
    distance_type = str(args.distance_type)
    reward_smooth_func = str(args.reward_smooth_func)
    l2_reg = args.l2_reg
    MINI_BATCH_SIZE = args.MINI_BATCH_SIZE
    rewardnet_layers = args.rewardnet_layers
    layer_dropout_rates = args.layer_dropout_rates
    goodbad_pairs_mode = int(args.goodbad_pairs_mode)

    # Define a dictionary mapping activation names to the actual functions
    activation_functions = {"tanh": tf.nn.tanh, "relu": tf.nn.relu, "leaky_relu": tf.nn.leaky_relu,}
    activations = [activation_functions[activation] for activation in args.rewardnet_activations]    # Convert the serialized strings to actual activation functions

    REWARD_THRESHOLD = max(D['exp_data'][settings.REWARD_COL])

    BC_Exp_trajectories = pd.read_csv('../c2_BC/BC_trajectories/BC_Exp_Trajectories.csv')
    BC_Opt_trajectories = pd.read_csv('../c2_BC/BC_trajectories/BC_OPT_Trajectories.csv')
    BC_NonOpt_trajectories = pd.read_csv('../c2_BC/BC_trajectories/BC_NonOPT_Trajectories.csv')
    SI_All_trajectories = pd.read_csv('../b1_Main/SI_Trajectories/SI_ALL_Trajectories.csv')
    Test_Trajecotires = utils.load_dict('../Data/processed_data_dict_subset1_updated_negselection_normalized.pkl')
    Test_Trajecotires_optml = utils.load_dict('../Data/processed_data_dict_subset1_updated_negselection_normalized.pkl')['test_df_optml']
    Test_Trajecotires_nonop = utils.load_dict('../Data/processed_data_dict_subset1_updated_negselection_normalized.pkl')['test_df_nonop']
    # Extracting survived and diseased trajectories
    Survvd_trajecotires = pd.concat([D['exp_df_survived'], D['train_policy_df_survived']])
    Deassd_trajecotires = pd.concat([D['exp_df_diseased'], D['train_policy_df_diseased']])

    # REMOVE TERMINAL STEPS
    values_to_remove = [REWARD_THRESHOLD, -REWARD_THRESHOLD]
    BC_Exp_trajectories = BC_Exp_trajectories[~BC_Exp_trajectories['reward'].isin(values_to_remove)]
    BC_Opt_trajectories = BC_Opt_trajectories[~BC_Opt_trajectories['reward'].isin(values_to_remove)]
    BC_NonOpt_trajectories = BC_NonOpt_trajectories[~BC_NonOpt_trajectories['reward'].isin(values_to_remove)]
    SI_All_trajectories = SI_All_trajectories[~SI_All_trajectories['reward'].isin(values_to_remove)]
    Test_Trajecotires_optml = Test_Trajecotires_optml[~Test_Trajecotires_optml['reward'].isin(values_to_remove)]
    Test_Trajecotires_nonop = Test_Trajecotires_nonop[~Test_Trajecotires_nonop['reward'].isin(values_to_remove)]
    Survvd_trajecotires = Survvd_trajecotires[~Survvd_trajecotires['reward'].isin(values_to_remove)]
    Deassd_trajecotires = Deassd_trajecotires[~Deassd_trajecotires['reward'].isin(values_to_remove)]

    BC_exp_data = prepare_data(BC_Exp_trajectories, state_cols, action_cols)
    BC_opt_data = prepare_data(BC_Opt_trajectories, state_cols, action_cols)
    BC_nonopt_data = prepare_data(BC_NonOpt_trajectories, state_cols, action_cols)
    SI_all_data = prepare_data(SI_All_trajectories, state_cols, action_cols)
    Test_optml_data = prepare_data(Test_Trajecotires_optml, state_cols, action_cols)
    Test_nonop_data = prepare_data(Test_Trajecotires_nonop, state_cols, action_cols)
    Survvd_data = prepare_data(Survvd_trajecotires, state_cols, action_cols)
    Deassd_data = prepare_data(Deassd_trajecotires, state_cols, action_cols)

    # creates a list of good, bad pairs to randomly select in each iteration
    good_bad_pairs = [[SI_all_data, BC_nonopt_data], [BC_opt_data, BC_nonopt_data], [Survvd_data, Deassd_data]]

    if goodbad_pairs_mode > 0:
        good_bad_pairs.append([BC_exp_data, BC_nonopt_data])
    if goodbad_pairs_mode > 1:
        good_bad_pairs.append([SI_all_data, BC_exp_data])
    if goodbad_pairs_mode > 2:
        good_bad_pairs.append([BC_opt_data, BC_exp_data])
    if goodbad_pairs_mode > 4:
        good_bad_pairs.append([SI_all_data, BC_exp_data])
    if goodbad_pairs_mode > 5:
        good_bad_pairs.append([SI_all_data, BC_opt_data])

    pair_len = len(good_bad_pairs)
    BATCH_SIZE = int(MINI_BATCH_SIZE * pair_len)
    print('Pair length: ' + str(pair_len))

    print(np.max(np.array(SI_All_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.max(np.array(BC_NonOpt_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.max(np.array(BC_Exp_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.max(np.array(BC_Opt_trajectories[D['UPDATED_STATE_COLs']])))

    print(np.min(np.array(SI_All_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.min(np.array(BC_NonOpt_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.min(np.array(BC_Exp_trajectories[D['UPDATED_STATE_COLs']])))
    print(np.min(np.array(BC_Opt_trajectories[D['UPDATED_STATE_COLs']])))

    # Removed SI vs optimal
    # Initialize the RewardNet and Model
    net = RewardNet(include_action=True, ob_dim=ob_dim_, ac_dim=ac_dim_, REWARD_THRESHOLD=REWARD_THRESHOLD, middle_layers=rewardnet_layers, dropout_rates=layer_dropout_rates, activations=activations, scale_factor=scale_factor, reward_smooth_func=reward_smooth_func)
    model = Model(net=net, batch_size=BATCH_SIZE, lr=lr, reward_distance_type=distance_type, ce_weight=ce_weight)

    # Create a session and initialize variables
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print("Iteration\tloss\taccuracy\tmean_good_r_per_traj\tmean_bad_r_per_traj\tmean_pairwise_loss_test_list\tmean_acc_test_list\tmean_r_per_good_traj_test_list\tmean_r_per_bad_traj_test_list")

    # Train the model
    for i in range(1, Epochs + 1):
        good_batch = random.choices(good_bad_pairs[0][0], k=MINI_BATCH_SIZE)
        bad_batch = random.choices(good_bad_pairs[0][1], k=MINI_BATCH_SIZE)

        x = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in good_batch], axis=0)
        y = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in bad_batch], axis=0)
        x_split = np.array([len(ob[0]) for ob in good_batch])
        y_split = np.array([len(ob[0]) for ob in bad_batch])

        for idx in range(1, pair_len):
            random_good_bad_pair = good_bad_pairs[idx]
            good_batch = random.choices(random_good_bad_pair[0], k=MINI_BATCH_SIZE)
            bad_batch = random.choices(random_good_bad_pair[1], k=MINI_BATCH_SIZE)

            x_ = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in good_batch], axis=0)
            y_ = np.concatenate([np.concatenate((a1, b1), axis=1) for (a1, b1) in bad_batch], axis=0)
            x_split_ = np.array([len(ob[0]) for ob in good_batch])
            y_split_ = np.array([len(ob[0]) for ob in bad_batch])

            x = np.concatenate((x, x_), axis=0)
            y = np.concatenate((y, y_), axis=0)
            x_split = np.concatenate((x_split, x_split_), axis=0)
            y_split = np.concatenate((y_split, y_split_), axis=0)

        feed_dict_ = {
            model.good_steps: x,
            model.bad_steps: y,
            model.good_traj_lengths: x_split,
            model.bad_traj_lengths: y_split,
            model.l: [0] * BATCH_SIZE,
            model.l2_reg: l2_reg}
        _, loss_, acc, v_good_trajs_means, v_bad_trajs_means = sess.run([model.update_op, model.loss, model.acc, model.v_good_trajs_means, model.v_bad_trajs_means], feed_dict=feed_dict_)

        # MANUALLY CHECK THE RETURN VALUES
        mean_r_per_good_traj = np.mean(v_good_trajs_means)
        mean_r_per_bad_traj = np.mean(v_bad_trajs_means)
        mean_pairwise_loss_test, mean_acc_test, mean_r_per_good_traj_test, mean_r_per_bad_traj_test = compute_test_results_subset(model, Test_optml_data, Test_nonop_data, BATCH_SIZE)
        print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(i, loss_, acc, mean_r_per_good_traj, mean_r_per_bad_traj, mean_pairwise_loss_test, mean_acc_test, mean_r_per_good_traj_test, mean_r_per_bad_traj_test))

    print(args)
    save_path = save_dir + f"/model_epoch_{i}.ckpt"
    saver.save(sess, save_path)
    save_args_file(args, save_dir + '/args_file.json')
    print('.................. Reward Model training and saving complete .................')
    print(f'.................. Model saved to {save_dir} .................')

