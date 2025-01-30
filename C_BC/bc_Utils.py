import copy
import pandas as pd
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
def create_trajectories(param, merged_df, bc_policy, state_cols, action_cols):
    df = copy.deepcopy(merged_df)
    df_param_match = df[df.type == param]
    df_param_other = df[df.type != param]

    # Replaces the m
    preds = bc_policy.predict(df_param_other[state_cols])
    pred_actions = np.argmax(preds, axis=1)
    one_hot_actions = one_hot(pred_actions, len(action_cols))

    df_param_other[action_cols] = one_hot_actions
    df_param_other.type = param + '_PRED'

    df_altered = pd.concat([df_param_match, df_param_other])
    df_altered = df_altered.sort_values(by=['stay_id', 'step_id'], ascending=[True, True])

    return df_altered