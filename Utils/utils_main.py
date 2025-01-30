import csv
from evaluation_graphs import *
import seaborn as sns
import a4_settings as a4_settings
import gc
import pickle as pk
import numpy as np
import pandas as pd

TRAIN_PHASE1 = "train_p1"
TRAIN_PHASE2 = "train_p2"

ori_state_columns = ['shock_index', 'age', 'gender', 'weight', 'readmission', 'sirs', 'elixhauser_vanwalraven', 'MechVent', 'heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH', 'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium',
             'ionized_calcium','albumin', 'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun', 'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total', 'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours', 'magnesium', 'bloc']

ORI_DATA_COLs_WO_TRAINTEST = []
ori_next_state_columns = ['next_' + x for x in ori_state_columns]
UPDATED_STATE_COLs = copy.deepcopy(ori_state_columns)
UPDATED_NEXT_STATE_COLs = copy.deepcopy(ori_next_state_columns)
ACTION_COLs = []
OPTIMALITY_SCORES_COL = 'optimality_score'

ori_action_columns = ['iv_fluids_quantile', 'vasopressors_quantile']
ori_moratality_reward_columns = ['done', 'mortality_hospital', 'ori_sofa_24hours', 'ori_lactate', 'next_ori_sofa_24hours', 'next_ori_lactate']
ori_optimality_score_columns = ['done', 'ori_mbp', 'next_ori_mbp', 'ori_lactate', 'next_ori_lactate', 'ori_sofa_24hours', 'next_ori_sofa_24hours']


# Testing Function
# Per sequence mean values
def test_per_sequence_mean_value(model, obs_epis_all):
    per_seq_mean_values = []
    for obs_seq_i in range(0, len(obs_epis_all)):
        per_sequence_values = model.get_state_value(obs_epis_all[obs_seq_i])
        per_seq_mean_values.append(np.mean(per_sequence_values))
    return per_seq_mean_values


# Per sequence mean values
def test_all_n_per_sequence_mean_values(model, obs_epis_all):
    per_seq_mean_values = []
    all_values = np.array([])
    for obs_seq_i in range(0, len(obs_epis_all)):

        per_sequence_values = model.get_state_value(obs_epis_all[obs_seq_i])
        seq_mean_value = np.mean(per_sequence_values)
        per_seq_mean_values.append(seq_mean_value)
        all_values = np.concatenate((all_values, per_sequence_values.flatten()))
    return np.mean(per_seq_mean_values), all_values


def q_values_all(sess, ac_size, phase, mainQN, discriminator, obs_, terminal_mortality_reward):

    q_values, pt_acs = sess.run((mainQN.q_outputs, mainQN.predicted_best_action), feed_dict={mainQN.state: obs_, mainQN.dropout_rate: 0.0})
    pt_acs_onehot = np.array([np.eye(ac_size)[x] for x in pt_acs])

    if phase == TRAIN_PHASE1:
        rewards = discriminator.get_reward_r1(np.concatenate([obs_, pt_acs_onehot], axis=1), 0.0)
    else:
        rewards = discriminator.get_reward_r2(np.concatenate([obs_, pt_acs_onehot], axis=1), terminal_mortality_reward, 0.0)

    return q_values, np.mean(q_values), rewards


def save_data_for_plots_v4(data_buffer, q_values_all, file_prefix_, gen_reward_col, iteration_idx):
    data_buffer = copy.deepcopy(data_buffer)
    q_cols = ['Q_' + str(x) for x in range(len(q_values_all[0]))]

    data_buffer[q_cols] = q_values_all
    file_output_all = file_prefix_ + '_output_all.csv'
    # Draw Figures
    run_eval_iter(gen_reward_col, data_buffer, "result/" + file_output_all.split('/')[-1], str(iteration_idx))

def draw_normalized_histo(data, no_bins, axis, color_, alpha_, label_):
    sns.histplot(data=data, stat="probability", discrete=True,  rug=True, ax=axis, color=color_, alpha=alpha_, label=label_)

def write_to_csv(outputs, file_graph_output):
    with open(file_graph_output, "w", newline='') as f:
        writer = csv.writer(f)
        for row in outputs:
            writer.writerow(row)


def create_sample_expert_trajectoris(expert_traj_size, ob_all, acs_all):
    random_indices = np.random.randint(0, len(acs_all), expert_traj_size)
    expert_trajectories = np.hstack([ob_all[random_indices], acs_all[random_indices]])
    return expert_trajectories


def convert_categorical_columns(data, column, column_name):
    # xxx = data[ori_state_columns]
    gender_one_hot_ = pd.get_dummies(data[column], column_name)
    columns = gender_one_hot_.columns

    return gender_one_hot_, columns


def get_one_hot_ary(original_array, no_actions):
    one_hot_ary = np.array([[0] * no_actions] * len(original_array))

    for i in range(len(one_hot_ary)):
        one_hot_ary[i][original_array[i]] = 1

    return one_hot_ary


def get_actions_one_hot(data):
    no_actions = len(data[ori_action_columns[0]].unique()) * len(data[ori_action_columns[1]].unique())
    original_array = np.array(data.apply(lambda x: int(x[ori_action_columns[0]] * 5 + x[ori_action_columns[1]] - 6), axis=1))
    one_hot_ary = get_one_hot_ary(original_array, no_actions)

    actions_one_hot_ = pd.DataFrame(one_hot_ary)
    actions_one_hot_.columns = ['action_one_hot_' + str(x) for x in range(no_actions)]

    return actions_one_hot_, actions_one_hot_.columns

def convert_data_columns(data):
    # CONVERT STATE COLUMNS
    # gender
    data.gender = data.gender.replace([0.5], 1).replace([-0.5], 0)
    state_gender_one_hot, state_gender_one_hot_columns = convert_categorical_columns(data, 'gender', 'gender_')
    data = pd.concat([data, state_gender_one_hot], axis=1)
    UPDATED_STATE_COLs.remove('gender')
    UPDATED_STATE_COLs.extend(state_gender_one_hot_columns)

    #   1.1.2 readmission
    data.readmission = data.readmission.replace([0.5], 1).replace([-0.5], 0)
    #   1.1.2 MechVent
    data.MechVent = data.MechVent.replace([0.5], 1).replace([-0.5], 0)

    # CONVERT NEXT STATE COLUMNS
    # gender
    data.next_gender = data.next_gender.replace([0.5], 1).replace([-0.5], 0)
    next_state_gender_one_hot, next_state_gender_one_hot_columns = convert_categorical_columns(data, 'next_gender', 'next_gender_')
    data = pd.concat([data, next_state_gender_one_hot], axis=1)
    UPDATED_NEXT_STATE_COLs.remove('next_gender')
    UPDATED_NEXT_STATE_COLs.extend(next_state_gender_one_hot_columns)

    # 1.1.2 readmission
    data.next_readmission = data.next_readmission.replace([0.5], 1).replace([-0.5], 0)
    # 1.1.2 MechVent
    data.next_MechVent = data.next_MechVent.replace([0.5], 1).replace([-0.5], 0)
    # 3. Append one hot encoded actions to the dataframe
    actions_one_hot, actions_one_hot_column_names = get_actions_one_hot(data)
    ACTION_COLs[:] = actions_one_hot_column_names
    data = pd.concat([data, actions_one_hot], axis=1)
    return data

# V2: Survived and Diseased Epi based mortality
def load_datav3(data, FINAL_REWARD_SCALE_, THRESHOLD_NONOP_PAIRS_):
    FINAL_REWARD_SCALE = FINAL_REWARD_SCALE_
    THRESHOLD_NONOP_PAIRS = THRESHOLD_NONOP_PAIRS_

    data = data[data['length'] > 1]
    data.reset_index(drop=True, inplace=True)
    data['output_total'] = data['output_total'].fillna(0)
    data['urine_output'] = data['urine_output'].fillna(0)
    # Got reward column
    data[a4_settings.REWARD_COL] = data[ori_moratality_reward_columns].apply(lambda row: a4_settings.reward_sepsis(row, FINAL_REWARD_SCALE), axis=1)
    # Got optimal and nonoptimal data
    df_optml, df_nonop = a4_settings.find_optimal_nonop_samples_compositeNclustered_approach(data, FINAL_REWARD_SCALE)
    # Added optimality Score
    stay_ids_all = list(data['stay_id'].unique())
    stay_ids_diseased = list(data['stay_id'][data['mortality_hospital'] == 1].unique())
    stay_ids_survived = list(data['stay_id'][data['mortality_hospital'] == 0].unique())

    df_survived = data[data['stay_id'].isin(stay_ids_survived)]
    df_diseased = data[data['stay_id'].isin(stay_ids_diseased)]

    df_survived = df_survived.reset_index(drop=True)
    df_diseased = df_diseased.reset_index(drop=True)
    df_optml = df_optml.reset_index(drop=True)
    df_nonop = df_nonop.reset_index(drop=True)

    return data, df_survived, df_diseased, df_optml, df_nonop

def save_dict(dict, dict_filename):
    with open(dict_filename, 'wb') as f:
        pk.dump(dict, f)

def load_dict(dict_filename):
    with open(dict_filename, 'rb') as f:
        loaded_dict = pk.load(f)
        return loaded_dict

def load_data_new(exp_policy_split_frac, FINAL_REWARD_SCALE, THRESHOLD_NONOP_PAIRS, dict_filename):
    # Load the initial dataset. Each row containing current and next state information.
    #  Contains the following columns  (note that all may not be needed, and use appropriately based on the disease):
    #  ['stay_id', 'step_id', 'time', 'ori_iv_fluids', 'ori_vasopressors', 'iv_fluids_level', 'vasopressors_level', 'mortality_hospital', 'mortality_90', 'start_offset', 'end_offset', 'iv_fluids', 'vasopressors', 'heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH', 'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun', 'creatinine', 'albumin', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total', 'magnesium', 'ionized_calcium', 'calcium', 'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours', 'age', 'gender', 'weight', 'readmission', 'elixhauser_vanwalraven', 'MechVent', 'sirs', 'shock_index', 'bloc', 'next_dbp', 'next_ptt', 'next_paco2', 'next_ionized_calcium', 'next_pH', 'next_inr', 'next_bicarbonate', 'next_shock_index', 'next_output_total', 'next_platelet', 'next_chloride', 'next_calcium', 'next_pao2fio2ratio', 'next_albumin', 'next_sirs', 'next_age', 'next_bilirubin_total', 'next_sofa_24hours', 'next_baseexcess', 'next_bloc', 'next_pao2', 'next_temperature', 'next_gender', 'next_creatinine', 'next_bun', 'next_elixhauser_vanwalraven', 'next_co2', 'next_magnesium', 'next_potassium', 'next_alt', 'next_sbp', 'next_urine_output', 'next_pt', 'next_mbp', 'next_respiratoryrate', 'next_glucose', 'next_sodium', 'next_MechVent', 'next_lactate', 'next_fio2', 'next_spo2', 'next_gcs', 'next_wbc', 'next_readmission', 'next_weight', 'next_heartrate', 'next_hemoglobin', 'next_ast', 'length', 'first_step_id', 'last_step_id', 'start', 'done', 'iv_fluids_quantile', 'vasopressors_quantile', 'train_test']
    data = pd.read_csv("../Data/<normalized_input_data_file.csv>")

    data[UPDATED_STATE_COLs] = data[UPDATED_STATE_COLs].fillna(0).apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    data[UPDATED_NEXT_STATE_COLs] = data[UPDATED_NEXT_STATE_COLs].fillna(0).apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    ori_cols_wo_traintest = copy.deepcopy(list(data.columns))
    ori_cols_wo_traintest.remove('train_test')

    ORI_DATA_COLs_WO_TRAINTEST[:] = ori_cols_wo_traintest
    data = convert_data_columns(data)

    train_data = data[data['train_test'] == "train"]
    test_data = data[data['train_test'] == "test"]
    total_train_users = len(train_data['stay_id'].unique())
    # Drop train_test columns
    train_data = train_data.drop(columns=['train_test'])
    test_data = test_data.drop(columns=['train_test'])

    all_train_indices = list(train_data.stay_id.unique())
    no_of_exp_policy_split = int(exp_policy_split_frac * total_train_users)
    exp_indices = random.sample(all_train_indices, no_of_exp_policy_split)
    policy_indices = list(set(all_train_indices).difference(exp_indices))

    exp_data = train_data[train_data['stay_id'].isin(exp_indices)]
    train_data_policy = train_data[train_data['stay_id'].isin(policy_indices)]
    test_data = test_data[test_data['stay_id'].isin(list(test_data.stay_id.unique()))]

    exp_data = exp_data.reset_index(drop=True)
    train_data_policy = train_data_policy.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    gc.collect()
    print('Retrieving exp data ...')
    exp_data_, exp_df_survived, exp_df_diseased, exp_df_optml, exp_df_nonop = load_datav3(exp_data, FINAL_REWARD_SCALE, THRESHOLD_NONOP_PAIRS)
    gc.collect()
    print('Retrieving train policy data ...')
    train_data_policy_, train_policy_df_survived, train_policy_df_diseased, train_policy_df_optml, train_policy_df_nonop = load_datav3(train_data_policy, FINAL_REWARD_SCALE, THRESHOLD_NONOP_PAIRS)
    gc.collect()
    print('Retrieving test data ...')
    test_data_, test_df_survived, test_df_diseased, test_df_optml, test_df_nonop = load_datav3(test_data, FINAL_REWARD_SCALE, THRESHOLD_NONOP_PAIRS)

    dict = {}
    dict['exp_data'] = exp_data_
    dict['exp_df_survived'] = exp_df_survived
    dict['exp_df_diseased'] = exp_df_diseased
    dict['exp_df_optml'] = exp_df_optml
    dict['exp_df_nonop'] = exp_df_nonop
    dict['train_data_policy_'] = train_data_policy_
    dict['train_policy_df_survived'] = train_policy_df_survived
    dict['train_policy_df_diseased'] = train_policy_df_diseased
    dict['train_policy_df_optml'] = train_policy_df_optml
    dict['train_policy_df_nonop'] = train_policy_df_nonop
    dict['test_data_'] = test_data_
    dict['test_df_survived'] = test_df_survived
    dict['test_df_diseased'] = test_df_diseased
    dict['test_df_optml'] = test_df_optml
    dict['test_df_nonop'] = test_df_nonop
    dict['UPDATED_STATE_COLs'] = UPDATED_STATE_COLs
    dict['ACTION_COLs'] = ACTION_COLs
    dict['UPDATED_NEXT_STATE_COLs'] = UPDATED_NEXT_STATE_COLs
    dict['ORI_DATA_COLs_WO_TRAINTEST'] = ORI_DATA_COLs_WO_TRAINTEST
    save_dict(dict, dict_filename)
    return exp_data_, exp_df_survived, exp_df_diseased, exp_df_optml, exp_df_nonop, train_data_policy_, train_policy_df_survived, train_policy_df_diseased, train_policy_df_optml, train_policy_df_nonop, test_data_, test_df_survived, test_df_diseased, test_df_optml, test_df_nonop

