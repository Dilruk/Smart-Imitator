import numpy as np
import pandas as pd
# no apache now
state_col = ['shock_index', 'age', 'gender', 'weight', 'readmission', 'sirs', 'elixhauser_vanwalraven',
             'MechVent',  'heartrate', 'respiratoryrate', 'spo2', 'temperature',
             'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH',
             'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium', 'ionized_calcium', 'albumin',
             'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun',
             'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total',
             'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours', 'magnesium', 'bloc']

# Created Next state columns programatically to maintain the same order
next_state_col = ['next_' + x for x in state_col]

# action_dis_col = ['iv_fluids_level', 'vasopressors_level']
action_dis_col = ['iv_fluids_quantile', 'vasopressors_quantile']
reward_cols = ['mortality_hospital', 'ori_sofa_24hours', 'next_ori_sofa_24hours', 'ori_lactate', 'next_ori_lactate']

REWARD_FUN = 'sepsis_composite_score'
REWARD_COL = 'reward'

SEED = 1
ITERATION_ROUND = 150 #80000
ACTION_SPACE = 25
BATCH_SIZE = 256
per_flag = True
per_alpha = 0.6     # PER hyperparameter
per_epsilon = 0.01  # PER hyperparameter
beta_start =0.9
# MODEL = 'DQN' # 'DQN' or 'FQI'
GAMMA = 0.99


def reward_sepsis(x, FINAL_REWARD_SCALE):
    res = 0
    if (x['done'] == 1 and x['mortality_hospital'] == 1):
        res = -FINAL_REWARD_SCALE
    elif (x['done'] == 1 and x['mortality_hospital'] == 0):
        res = FINAL_REWARD_SCALE
    elif x['done'] == 0:
        if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
            res += -0.025 #C0
        res += -0.125 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
        res += -2 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
    else:
        res = np.nan
    return res


# Higher the worse
def sepsis_composite_score(x):
    result = 0

    if ((x['next_ori_lactate']>x['ori_lactate']) & (x['next_ori_lactate']>2)):
        result += 1

    if(x['next_ori_sofa_24hours']>x['ori_sofa_24hours']):
        result += 1

    if(x['next_ori_mbp']<65 and x['next_ori_mbp']<x['ori_mbp']):
        result += 1
    return result


def find_optimal_nonop_samples(df):
    df['composite_score'] = df.apply(eval('sepsis_composite_score'), axis = 1)
    df['previous_score'] = df.groupby('stay_id')['composite_score'].shift(1).fillna(4)
    df['previous_score'] = pd.to_numeric(df['previous_score'], downcast='integer')
    df['negative_samples'] = df.apply(lambda x: 1 if x['composite_score']>x['previous_score'] else 0, axis = 1)

    df_optml = df[df['negative_samples'] == 0]
    df_optml = df_optml.reset_index(drop=True)

    df_nonop = df[df['negative_samples'] == 1]
    df_nonop = df_nonop.reset_index(drop=True)

    return df_optml, df_nonop

# def reward_only_long(x):
#     res = 0
#     if (x['done'] == 1 and x['mortality_hospital'] == 1):
#         res += -15
#     elif (x['done'] == 1 and x['mortality_hospital'] == 0):
#         res += 15
#     elif x['done'] == 0:
#         res = 0
#     else:
#         res = np.nan
#     return res
#
# def reward_mortality_sofa_lactate(x):
#     res = 0
#     if (x['done'] == 1 and x['mortality_hospital'] == 1):
#         res = -15
#     elif (x['done'] == 1 and x['mortality_hospital'] == 0):
#         res = 15
#     elif x['done'] == 0:
#         if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
#             res += -0.025 #C0
#         res += -0.125 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
#         res += -2 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
#     else:
#         res = np.nan
#     return res
#
# def reward_mortality_sofa_lactate2(x):
#     res = 0
#     if (x['done'] == 1 and x['mortality_hospital'] == 1):
#         res = -100
#     elif (x['done'] == 1 and x['mortality_hospital'] == 0):
#         res = 100
#     elif x['done'] == 0:
#         if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
#             res += -0.025 #C0
#         res += -0.125 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
#         res += -2 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
#     else:
#         res = np.nan
#     return res
#
# def score_mortality_sofa_lactate_mbp(x):
#     result = 0
#     if x['done'] == 0:
#         if (x['next_ori_lactate']) <2:
#             result += 2
#         elif (x['next_ori_lactate']) <4:
#             result +=1
#         elif (x['next_ori_lactate']<x['ori_lactate']):
#             result += 0.5
#         else:
#             result -= 1
#
#         if(x['next_ori_sofa_24hours']<x['ori_sofa_24hours']):
#             result += 2
#
#         if(x['next_ori_mbp']<=80 and x['next_ori_mbp']>=70):
#             result += 2
#     else:
#         result = np.nan
#     return result
