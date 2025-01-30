import copy
import numpy as np


def check_for_intermittent_opt_nonopt_steps(merged_df):
    test = merged_df[['stay_id', 'step_id', 'time', 'type', 'cluster_negative', 'negative_samples']]
    counts = test.groupby('stay_id')['type'].nunique()
    # check if any counts are greater than 1
    result = any(counts > 1)
    # print the result
    return result


def one_hot(pt_acs_all_batch, ac_size):
    return np.array([np.eye(ac_size)[x] for x in pt_acs_all_batch])


# This method predicts actions (one hot) based on the passed model and replaces the original (experts provided) action.
# For example, when using the BC model for EXP, we use the original instances identified as EXP unchanged while the actions for the rest are changed based on the learnt BC_Exp model.
def create_trajectories(sess, param, merged_df, policy, state_cols, action_cols):
    df_altered = copy.deepcopy(merged_df)

    # Replaces the m
    pred_actions = sess.run(policy.mainQN.predicted_best_action, feed_dict={policy.mainQN.state: merged_df[state_cols], policy.mainQN.dropout_rate: 0.0})
    one_hot_actions = one_hot(pred_actions, len(action_cols))

    df_altered[action_cols] = one_hot_actions
    df_altered.type = param + '_PRED'

    df_altered = df_altered.sort_values(by=['stay_id', 'step_id'], ascending=[True, True])

    return df_altered