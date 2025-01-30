# This file
import Utils.utils_main as Utils

if __name__ == '__main__':
    # File to save the processed dataset.
    dict_filename = '../Data/<data_file>.pkl'

    exp_policy_split_frac = 0.3     # Ratio to split the dataset to use as retrospective clinician behaviour data.
    FINAL_REWARD_SCALE = 15         # Reward scale
    THRESHOLD_NONOP_PAIRS = 0       # Threshold to determine the non-optimality
    Utils.load_data_new(exp_policy_split_frac, FINAL_REWARD_SCALE, THRESHOLD_NONOP_PAIRS, dict_filename)
    print('done')
