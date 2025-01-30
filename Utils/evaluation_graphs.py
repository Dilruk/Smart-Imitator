# -*- coding: utf-8 -*-
"""
Created on  Jan 14 2021

@author: Siqi Liu
"""
# TODO: Allows iteration wise graph generation
import sys
sys.path.append("../Utils")
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
# import Utils.setting_4 as setting_4
import settings as setting_4
import copy
from functools import reduce
from sklearn.metrics import jaccard_score


def run_eval_iter(reward_col, data, perfix, iteration_idx):
    #     def plot_loss(loss):
    #         if loss:
    #             plt.figure(figsize=(7,4))
    #             plt.plot(loss)
    #             plt.savefig(res_dir + 'loss.png',dpi = 100)

    def tag_conc_rate_and_diff_mean(dt):
        for v in action_types:
            dt[v + '_conc_rate'] = dt[v + '_conc'].mean()
            dt[v + '_diff_mean'] = dt[v + '_diff'].mean()
        return dt

    def discre_conc_level(x):
        xx = [0.1, 0.3, 0.5, 0.7, 0.9]
        for t in xx:
            if x >= t - 0.1 and x < t + 0.1:
                return t
        if x == 1:
            return 0.9

    def discre_diff_level(x):
        xx = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        for t in xx:
            if x >= t - 0.5 and x < t + 0.5:
                return t
        if x == 4:
            return 4

    def q_vs_outcome(data, outcome_col1='mortality_hospital'):
        q_vals_phys = data.apply(lambda x: x['Q_' + str(int(x[phys_action]))], axis=1)
        pp = pd.Series(q_vals_phys)
        phys_df = pd.DataFrame(pp)
        phys_df['mort'] = copy.deepcopy(np.array(data[outcome_col1]))

        bin_medians = []
        mort = []
        mort_std = []
        # i = -15
        # while i <= 20:
        # The function sets a range of quantiles (from 0.01 to 0.99) of the Q-values, and in this range, it loops through each quantile value with a step size of 0.05 (i += 0.05).
        # For each step, it defines a bin of Q-values within +/-0.5 of the current quantile value. It then computes the mortality rate within that bin as the sum of 'mort' values divided by the number of entries in that bin.
        # If the bin has at least 2 entries, it records the median Q-value, mortality rate, and standard error of the mortality rate in their respective lists.
        # If there are no entries in the bin (ZeroDivisionError), it passes.
        i = q_vals_phys.quantile([0.01, 0.99]).values[0]    # returns the 1st percentile of the q_vals
        while (i <= q_vals_phys.quantile([0.01, 0.99]).values[1]):  # while 1st percentile is less than the 99th percentile
            count = phys_df.loc[(phys_df[0] > i - 0.5) & (phys_df[0] < i + 0.5)]    #
            try:
                res = sum(count['mort']) / float(len(count))
                if len(count) >= 2:
                    bin_medians.append(i)
                    mort.append(res)
                    mort_std.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += 0.05
        q_value = pd.DataFrame(bin_medians, columns=["q_value"])
        med = pd.DataFrame(sliding_mean(mort, window=1), columns=["med"])
        ci = pd.DataFrame(sliding_mean(mort_std, window=1), columns=["std"])
        bb = pd.concat([med, ci, q_value], ignore_index=True, axis=1)

        res_dt = pd.DataFrame(bb)
        res_dt.reset_index()

        return res_dt

    def quantitive_eval(data, res_dt, outcome_col='mortality_hospital'):
        data['Random_action'] = [random.randrange(action_num) for i in range(len(data))]
        cc = data[phys_action].value_counts()
        data['One-size-fit-all_action'] = cc.index[np.argmax(cc)]

        data['Random_Q'] = data.apply(lambda x: x['Q_' + str(int(x['Random_action']))], axis=1)
        data['One-size-fit-all_Q'] = data.apply(lambda x: x['Q_' + str(int(x['One-size-fit-all_action']))], axis=1)

        # maybe change to doubly robust estimation / WIS
        q_dr_dt = pd.DataFrame()
        for mod in ['ai', 'phys', 'Random', 'One-size-fit-all']:
            q_dr_dt.loc[mod, 'Q'] = data[mod + '_Q'].mean()

        def find_nearest_Q(Q_mean, res_dt):
            ind = np.argmin([abs(Q_mean - i) for i in res_dt[2]])
            Q_res = res_dt.index[ind]
            return Q_res

        for mod in ['ai', 'phys', 'Random', 'One-size-fit-all']:
            q_dr_dt.loc[mod, 'mortality'] = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod, 'Q'], res_dt), 0]
            q_dr_dt.loc[mod, 'std'] = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod, 'Q'], res_dt), 1]

        #         data = data[data['mortality_hospital']==0]
        #         data=data.reset_index()
        v_cwpdis, ess = cwpdis_ess_eval(data)
        v_WIS = cal_WIS(data)
        v_phys = cal_phy_V(data)

        jaccard_group = data.groupby('stay_id')
        jaccard_df_micro = jaccard_group.apply(calc_jaccard_micro).reset_index()
        #         jaccard_df_macro = jaccard_group.apply(calc_jaccard_macro).reset_index()
        micro_jacc = jaccard_df_micro.iloc[:, 1].mean()
        #         print(micro_jacc)
        #         macro_jacc = jaccard_df_macro.iloc[:, 1].mean()
        CWPDIS_phys, ess_phys = cwpdis_ess_eval_phys(data)
        #         WIS_phys = cal_WIS_phys(data)
        q_dr_dt.loc['ai', 'v_WIS'] = v_WIS
        q_dr_dt.loc['ai', 'v_CWPDIS'] = v_cwpdis
        q_dr_dt.loc['ai', 'effective_sample_size'] = ess
        #         q_dr_dt.loc['ai', 'jaccard_score'] = micro_jacc

        q_dr_dt.loc['phys', 'v_WIS'] = v_phys
        q_dr_dt.loc['phys', 'v_CWPDIS'] = CWPDIS_phys
        q_dr_dt.loc['phys', 'effective_sample_size'] = ess_phys
        q_dr_dt.loc['phys', 'jaccard_score'] = micro_jacc

        q_dr_dt.to_csv(res_dir + 'iter_' + iteration_idx + '_qmean_and_deathreachrate.csv', encoding='gb18030')

        return q_dr_dt

    def action_concordant_rate(data):
        conc_dt = pd.DataFrame()
        for i, v in enumerate(action_types):
            phys_col = v + '_level'
            ai_col = v + '_level_ai'
            conc_dt.loc[v, 'concordant_rate'] = str(round(np.mean(data[phys_col] == data[ai_col]) * 100, 1)) + '%'

        conc_dt.loc['all', 'concordant_rate'] = str(round(np.mean(data[phys_action] == data[ai_action]) * 100, 1)) + '%'

        #         v_cwpdis, ess = cwpdis_ess_eval(data)
        #         v_WIS = cal_WIS(data)
        #         conc_dt.loc['all', 'WIS'] = v_WIS
        #         conc_dt.loc['all', 'v_cwpdis'] = v_cwpdis
        #         conc_dt.loc['all', 'ess'] = ess

        conc_dt.to_csv(res_dir + 'iter_' + iteration_idx + '_action_concordant_rate.csv', encoding='gb18030')

        return conc_dt

    def sliding_mean(data_array, window=1):
        new_list = []
        for i in range(len(data_array)):
            indices = range(max(i - window + 1, 0),
                            min(i + window + 1, len(data_array)))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)
        return np.array(new_list)

    def make_df_diff(op_actions, df_in):
        op_vaso_med = []
        op_iv_med = []
        for action in op_actions:
            iv, vaso = inv_action_map[action]
            op_vaso_med.append(vaso_vals[vaso])
            op_iv_med.append(iv_vals[iv])
        iv_diff = np.array(op_iv_med) - np.array(df_in['ori_iv_fluids'])
        vaso_diff = np.array(op_vaso_med) - np.array(df_in['ori_vasopressors'])
        df_diff = pd.DataFrame()
        df_diff['mort'] = np.array(df_in['mortality_hospital'])
        df_diff['iv_diff'] = iv_diff
        df_diff['vaso_diff'] = vaso_diff
        return df_diff

    def cwpdis_ess_eval(data):
        df = data.copy()
        df = df.sort_values(by=['stay_id', 'step_id'])

        df['step_id'] = pd.to_numeric(df['step_id'], downcast='integer')
        # data.head()
        # data['concordant'] = [random.randint(0,1) for i in range(len(data))]
        df['concordant'] = df.apply(lambda x: (x[phys_action] == x[ai_action]) + 0, axis=1)

        # tag pnt
        df = df.groupby('stay_id').apply(cal_pnt)
        v_cwpdis = 0
        for t in range(1, max(df['step_id']) + 1):
            tmp = df[df['step_id'] == t - 1]
            if sum(tmp['pnt']) > 0:
                v_cwpdis += setting_4.GAMMA ** t * (sum(tmp[reward_col] * tmp['pnt']) / sum(tmp['pnt']))

        ess = sum(df['pnt'])

        return v_cwpdis, ess

    def cwpdis_ess_eval_phys(data):
        df = data.copy()
        df = df.sort_values(by=['stay_id', 'step_id'])

        df['step_id'] = pd.to_numeric(df['step_id'], downcast='integer')
        # data.head()
        # data['concordant'] = [random.randint(0,1) for i in range(len(data))]
        df['concordant'] = df.apply(lambda x: (x[phys_action] == x[phys_action]) + 0, axis=1)

        # tag pnt
        df = df.groupby('stay_id').apply(cal_pnt)
        v_cwpdis = 0
        for t in range(1, max(df['step_id']) + 1):
            tmp = df[df['step_id'] == t - 1]
            if sum(tmp['pnt']) > 0:
                v_cwpdis += setting_4.GAMMA ** t * (sum(tmp[reward_col] * tmp['pnt']) / sum(tmp['pnt']))

        ess = sum(df['pnt'])
        return v_cwpdis, ess

    def cal_pnt(dt):
        dt['conc_cumsum'] = dt['concordant'].cumsum()
        dt['pnt'] = (dt['conc_cumsum'] == (dt['step_id'] + 1)) + 0
        return dt

    def calculate_p1t(data, gamma=0.9):
        dt = data.copy()
        gamma = setting_4.GAMMA
        #         dt = dt.reset_index(drop=True)
        #         dt['cur_index'] = dt.index
        dt['p1_'] = dt.apply(lambda x: 1 if x['is_equal'] == 1 else 0, axis=1)
        dt['p1H_'] = reduce(lambda y, z: y * z, dt['p1_'].tolist())
        dt['gamma_rt'] = dt.apply(lambda x: gamma ** (x['step_id']) * x[reward_col], axis=1)
        dt['sum_gamma_rt'] = sum(dt['gamma_rt'])
        dt = dt.loc[max(dt.index)]
        return dt

    def cal_WIS(data):
        df = data.copy()
        df = df.reset_index(drop=True)
        df['is_equal'] = df.apply(lambda x: (x[phys_action] == x[ai_action]) + 0, axis=1)
        #         df = df.sort_values(by = ['stay_id','step_id'])
        tmp_df = df.groupby('stay_id').apply(calculate_p1t)
        D = len(df.stay_id.unique())
        wH = sum(tmp_df['p1H_']) / D
        tmp_df['Vwis'] = tmp_df.apply(lambda x: x['p1H_'] / wH * x['sum_gamma_rt'] if wH * x['sum_gamma_rt'] != 0 else 0, axis=1)
        WIS = sum(tmp_df['Vwis']) / D
        return WIS

    def cal_WIS_phys(data):
        df = data.copy()
        df = df.reset_index(drop=True)
        df['is_equal'] = df.apply(lambda x: (x[phys_action] == x[phys_action]) + 0, axis=1)
        df = df.sort_values(by=['stay_id', 'step_id'])
        tmp_df = df.groupby('stay_id').apply(calculate_p1t)
        D = len(df.stay_id.unique())
        wH = sum(tmp_df['p1H_']) / D
        tmp_df['Vwis'] = tmp_df.apply(lambda x: x['p1H_'] / wH * x['sum_gamma_rt'] if wH * x['sum_gamma_rt'] != 0 else 0, axis=1)
        WIS = sum(tmp_df['Vwis']) / D
        return WIS

    def cal_phy_V(data):
        df = data.copy()
        phys_vals = []
        unique_ids = df['stay_id'].unique()
        for uid in unique_ids:
            traj = df.loc[df['stay_id'] == uid]
            ret = 0
            reversed_traj = traj.iloc[::-1]
            for row in reversed_traj.index:
                ret = reversed_traj.loc[row, reward_col] + setting_4.GAMMA * ret
            #             if ret >30 or ret <-30:
            #                 continue
            phys_vals.append(ret)
        return np.mean(phys_vals)

    def calc_jaccard_macro(stay):
        phys_action = stay['phys_action'].values
        rl_action = stay['ai_action'].values
        jaccard_i = jaccard_score(phys_action, rl_action, average='macro')
        return pd.Series(jaccard_i)

    def calc_jaccard_micro(stay):
        phys_action = stay['phys_action'].values
        rl_action = stay['ai_action'].values
        jaccard_i = jaccard_score(phys_action, rl_action, average='micro')
        return pd.Series(jaccard_i)


    def make_heat_map_plots(actions_low_tuple=None, actions_mid_tuple=None, actions_high_tuple=None, graph_name=None):
        actions_low_tuple = np.array(actions_low_tuple)
        actions_mid_tuple = np.array(actions_mid_tuple)
        actions_high_tuple = np.array(actions_high_tuple)

        # In[17]:
        # DRAW: Draws the action heatmaps for low medium and high sofa levels.
        actions_low_iv = actions_low_tuple[:, 0]
        actions_low_vaso = actions_low_tuple[:, 1]
        hist_low, x_edges, y_edges = np.histogram2d(actions_low_iv, actions_low_vaso, bins=5)

        actions_mid_iv = actions_mid_tuple[:, 0]
        actions_mid_vaso = actions_mid_tuple[:, 1]
        hist_mid, _, _ = np.histogram2d(actions_mid_iv, actions_mid_vaso, bins=5)

        actions_high_iv = actions_high_tuple[:, 0]
        phys_actions_high_vaso = actions_high_tuple[:, 1]
        hist_high, _, _ = np.histogram2d(actions_high_iv, phys_actions_high_vaso, bins=5)

        # Create plots without the domination (0,0) cell to make the rest more visible
        hist_low_without_00 = np.copy(hist_low)
        hist_mid_without_00 = np.copy(hist_mid)
        hist_high_without_00 = np.copy(hist_high)
        hist_low_without_00[0, 0] = 0
        hist_mid_without_00[0, 0] = 0
        hist_high_without_00[0, 0] = 0

        x_edges = np.arange(-0.5, 5)
        y_edges = np.arange(-0.5, 5)
        big_size = 20
        BIGGER_SIZE = 20

        # Plotting with (0,0) cell
        f, axs = plt.subplots(1, 3, figsize=(16, 4))
        titles = ["Low SOFA policy", "Mid SOFA policy", "High SOFA policy"]
        data = [hist_low, hist_mid, hist_high]
        for ax, hist, title in zip(axs, data, titles):
            im = ax.pcolormesh(x_edges, y_edges, hist, cmap='Blues' if title == "Low SOFA policy" else ('Greens' if title == "Mid SOFA policy" else 'OrRd'))
            ax.set_title(f"{title} - {graph_name}", fontsize=big_size)
            ax.set_xlabel('Vasopressor dose', fontsize=big_size)
            ax.set_ylabel('IV fluid dose', fontsize=big_size)
            f.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{res_dir}iter_{iteration_idx}_{graph_name}_Action_Heatmap_with_00.png', dpi=300)
        plt.close(f)

        # Plotting without (0,0) cell
        f, axs = plt.subplots(1, 3, figsize=(16, 4))
        data_without_00 = [hist_low_without_00, hist_mid_without_00, hist_high_without_00]
        for ax, hist, title in zip(axs, data_without_00, titles):
            im = ax.pcolormesh(x_edges, y_edges, hist, cmap='Blues' if title == "Low SOFA policy" else ('Greens' if title == "Mid SOFA policy" else 'OrRd'))
            ax.set_title(f"{title} (w/o cell 0) - {graph_name}", fontsize=big_size)
            ax.set_xlabel('Vasopressor dose', fontsize=big_size)
            ax.set_ylabel('IV fluid dose', fontsize=big_size)
            f.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{res_dir}iter_{iteration_idx}_{graph_name}_Action_Heatmap_without_00.png', dpi=300)
        plt.close(f)
        # In[21]:

        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        # f, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(16, 4))
        #
        # ax1.imshow(np.flipud(hist_low), cmap="Blues", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        # ax2.imshow(np.flipud(hist_mid), cmap="OrRd", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        # ax3.imshow(np.flipud(hist_high), cmap="Greens", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        #
        # # Major ticks
        # ax1.set_xticks(np.arange(0, 5, 1))
        # ax1.set_yticks(np.arange(0, 5, 1))
        # ax2.set_xticks(np.arange(0, 5, 1))
        # ax2.set_yticks(np.arange(0, 5, 1))
        # ax3.set_xticks(np.arange(0, 5, 1))
        # ax3.set_yticks(np.arange(0, 5, 1))
        #
        # # Labels for major ticks
        # ax1.set_xticklabels(np.arange(0, 5, 1))
        # ax1.set_yticklabels(np.arange(0, 5, 1))
        # ax2.set_xticklabels(np.arange(0, 5, 1))
        # ax2.set_yticklabels(np.arange(0, 5, 1))
        # ax3.set_xticklabels(np.arange(0, 5, 1))
        # ax3.set_yticklabels(np.arange(0, 5, 1))
        #
        # # Minor ticks
        # ax1.set_xticks(np.arange(-.5, 5, 1), minor=True)
        # ax1.set_yticks(np.arange(-.5, 5, 1), minor=True)
        # ax2.set_xticks(np.arange(-.5, 5, 1), minor=True)
        # ax2.set_yticks(np.arange(-.5, 5, 1), minor=True)
        # ax3.set_xticks(np.arange(-.5, 5, 1), minor=True)
        # ax3.set_yticks(np.arange(-.5, 5, 1), minor=True)
        #
        # # Gridlines based on minor ticks
        # ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
        # ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
        # ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)
        #
        # im1 = ax1.pcolormesh(x_edges, y_edges, hist_low, cmap='Blues')
        # im2 = ax2.pcolormesh(x_edges, y_edges, hist_mid, cmap='Greens')
        # im3 = ax3.pcolormesh(x_edges, y_edges, hist_high, cmap='OrRd')
        # im4 = ax4.pcolormesh(x_edges, y_edges, hist_low_without_00, cmap='Blues')
        # im5 = ax5.pcolormesh(x_edges, y_edges, hist_mid_without_00, cmap='Greens')
        # im6 = ax6.pcolormesh(x_edges, y_edges, hist_high_without_00, cmap='OrRd')
        # f.colorbar(im1, ax=ax1)
        # f.colorbar(im2, ax=ax2)
        # f.colorbar(im3, ax=ax3)
        # f.colorbar(im4, ax=ax4)
        # f.colorbar(im5, ax=ax5)
        # f.colorbar(im6, ax=ax6)
        #
        # ax1.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax2.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax3.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax1.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax2.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax3.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax1.set_title("Low SOFA policy - " + graph_name, fontsize=big_size)
        # ax2.set_title("Mid SOFA policy - " + graph_name, fontsize=big_size)
        # ax3.set_title("High SOFA policy - " + graph_name, fontsize=big_size)
        #
        # for tick in ax1.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax2.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax3.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax1.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax2.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax3.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # plt.tight_layout()
        # plt.savefig(res_dir + 'iter_' + iteration_idx + '_' + graph_name + '_Action_Heatmap.png', dpi=300)
        #
        # ax4.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax5.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax6.set_ylabel('IV fluid dose', fontsize=big_size)
        # ax4.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax5.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax6.set_xlabel('Vasopressor dose', fontsize=big_size)
        # ax4.set_title("Low SOFA policy (w/o cell 0) - " + graph_name, fontsize=big_size)
        # ax5.set_title("Mid SOFA policy (w/o cell 0) - " + graph_name, fontsize=big_size)
        # ax6.set_title("High SOFA policy (w/o cell 0) - " + graph_name, fontsize=big_size)
        #
        # for tick in ax4.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax5.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax6.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax4.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax5.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        # for tick in ax6.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(BIGGER_SIZE)
        #
        # plt.tight_layout()
        # plt.savefig(res_dir + 'iter_' + iteration_idx + '_' + graph_name + '_Action_Heatmap_(wo cell 0).png', dpi=300)

    # main starts here

    # prepare for data
    np.random.seed(523)
    random.seed(523)
    action_num = 25
    Q_list = ['Q_' + str(i) for i in range(action_num)]
    action_types = ['iv_fluids', 'vasopressors']
    phys_Q = 'phys_Q'
    phys_action = 'phys_action'     # Clinician recommended Action
    ai_action = 'ai_action'         # AI recommended Action

    # data[action_types] = data[action_types].astype(int)
    data[action_types[0] + '_quantile'] = data[action_types[0] + '_quantile'].astype(int)
    data[action_types[1] + '_quantile'] = data[action_types[1] + '_quantile'].astype(int)
    if phys_action not in data.columns.tolist():
        data[phys_action] = data.apply(lambda x: int(x[action_types[0] + '_quantile'] * 5 + x[action_types[1] + '_quantile'] - 6), axis=1)
    data['ai_Q'] = np.max(data[Q_list], axis=1)                                     # best Q value picked by AI
    data[ai_action] = np.argmax(np.array(data[Q_list]), axis=1)                     # best action picked by AI
    data[phys_Q] = data.apply(lambda x: x['Q_' + str(int(x[phys_action]))], axis=1) # best Q value picked by Clinician
    # AI action is divided to iv and vaso bins
    data[action_types[0] + '_level_ai'] = (data[ai_action] / 5 + 1).apply(lambda x: int(x))
    data[action_types[1] + '_level_ai'] = (data[ai_action] % 5 + 1).apply(lambda x: int(x))
    # compute the diff and concordances between AI and Cliincian actions. Resulting conc column has 0 and 1, indicating non concordance and concordance.
    for v in action_types:
        data[v + '_diff'] = data[v + '_level'] - data[v + '_level_ai']
        data[v + '_conc'] = (data[v + '_level'] == data[v + '_level_ai']) + 0

    data = data.groupby('stay_id').apply(tag_conc_rate_and_diff_mean)
    # Computes the discrete difference and concordance levels between AI and Clinician actions per each action (vaso and iv).
    for v in action_types:
        data[v + '_diff_mean_level'] = data[v + '_diff_mean'].apply(discre_diff_level)
        data[v + '_conc_rate_level'] = data[v + '_conc_rate'].apply(discre_conc_level)

    data = data.reset_index(drop=True).copy()

    # Create dir for result

    res_dir = perfix + '/'

    if os.path.isdir(res_dir) == False:
        os.makedirs(res_dir)

        # plots of actions_difference vs mortality

    # In[7]:

    interventions = data[['ori_vasopressors', 'ori_iv_fluids']]

    # In[8]:
    # Removes any negative dose values recorded. Make them 0s (i gs)
    adjusted_vaso = interventions["ori_vasopressors"][interventions["ori_vasopressors"] > 0]
    adjusted_iv = interventions["ori_iv_fluids"][interventions["ori_iv_fluids"] > 0]

    # In[9]:
    # Compute the dose quantile borders using the original Clinician doses
    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.125, 0.375, 0.625, 0.875]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.125, 0.375, 0.625, 0.875]))

    # In[10]:

    #     data['deeprl2_actions'] = (data['iv_fluids_level_ai']-1)*5 + data['vasopressors_level_ai']-1
    data['deeprl2_actions'] = data['ai_action']
    data['phys_actions'] = data['iv_fluids_quantile'] * 5 + data['vasopressors_quantile'] - 6

    # In[12]:
    # divide the data into Low, Medium and High groups
    # Low SOFA
    df_test_orig_low = data[data['ori_sofa_24hours'] <= 5]

    # # Middling SOFA
    df_test_orig_mid = data[data['ori_sofa_24hours'] > 5]
    df_test_orig_mid = df_test_orig_mid[df_test_orig_mid['ori_sofa_24hours'] < 15]

    # # High SOFA
    df_test_orig_high = data[data['ori_sofa_24hours'] >= 15]

    # In[13]:

    # Now re-select the phys_actions, autoencode_actions, and deeprl2_actions based on the statified dataset
    deeprl2_actions_low = df_test_orig_low['deeprl2_actions'].values
    phys_actions_low = df_test_orig_low['phys_actions'].values

    deeprl2_actions_mid = df_test_orig_mid['deeprl2_actions'].values
    phys_actions_mid = df_test_orig_mid['phys_actions'].values

    deeprl2_actions_high = df_test_orig_high['deeprl2_actions'].values
    phys_actions_high = df_test_orig_high['phys_actions'].values

    # In[14]:

    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i, j]
            count += 1

    # In[15]:
    # For each severity group, computes the Clinician (phys_action) and AI (deeprl2) actions in to an array of 2, indicating 5 x 5.
    phys_actions_low_tuple = [None for i in range(len(phys_actions_low))]
    deeprl2_actions_low_tuple = [None for i in range(len(phys_actions_low))]

    phys_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]
    deeprl2_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]

    phys_actions_high_tuple = [None for i in range(len(phys_actions_high))]
    deeprl2_actions_high_tuple = [None for i in range(len(phys_actions_high))]

    for i in range(len(phys_actions_low)):
        phys_actions_low_tuple[i] = inv_action_map[phys_actions_low[i]]
        deeprl2_actions_low_tuple[i] = inv_action_map[deeprl2_actions_low[i]]

    for i in range(len(phys_actions_mid)):
        phys_actions_mid_tuple[i] = inv_action_map[phys_actions_mid[i]]
        deeprl2_actions_mid_tuple[i] = inv_action_map[deeprl2_actions_mid[i]]

    for i in range(len(phys_actions_high)):
        phys_actions_high_tuple[i] = inv_action_map[phys_actions_high[i]]
        deeprl2_actions_high_tuple[i] = inv_action_map[deeprl2_actions_high[i]]

    # In[16]:
    make_heat_map_plots(phys_actions_low_tuple, phys_actions_mid_tuple, phys_actions_high_tuple, 'Clinician')
    make_heat_map_plots(deeprl2_actions_low_tuple, deeprl2_actions_mid_tuple, deeprl2_actions_high_tuple, 'RL')

    # phys_actions_low_tuple = np.array(phys_actions_low_tuple)
    # deeprl2_actions_low_tuple = np.array(deeprl2_actions_low_tuple)
    #
    # phys_actions_mid_tuple = np.array(phys_actions_mid_tuple)
    # deeprl2_actions_mid_tuple = np.array(deeprl2_actions_mid_tuple)
    #
    # phys_actions_high_tuple = np.array(phys_actions_high_tuple)
    # deeprl2_actions_high_tuple = np.array(deeprl2_actions_high_tuple)
    #
    #
    # make_df_diff()
    # # In[17]:
    # # DRAW: Draws the action heatmaps for low medium and high sofa levels for Clinicians.
    # phys_actions_low_iv = phys_actions_low_tuple[:, 0]
    # phys_actions_low_vaso = phys_actions_low_tuple[:, 1]
    # hist_ph1, x_edges, y_edges = np.histogram2d(phys_actions_low_iv, phys_actions_low_vaso, bins=5)
    #
    # phys_actions_mid_iv = phys_actions_mid_tuple[:, 0]
    # phys_actions_mid_vaso = phys_actions_mid_tuple[:, 1]
    # hist_ph2, _, _ = np.histogram2d(phys_actions_mid_iv, phys_actions_mid_vaso, bins=5)
    #
    # phys_actions_high_iv = phys_actions_high_tuple[:, 0]
    # phys_actions_high_vaso = phys_actions_high_tuple[:, 1]
    # hist_ph3, _, _ = np.histogram2d(phys_actions_high_iv, phys_actions_high_vaso, bins=5)
    #
    # # Create plots without the domination (0,0) cell to make the rest more visible
    # hist_ph1_without_00 = np.copy(hist_ph1)
    # hist_ph2_without_00 = np.copy(hist_ph2)
    # hist_ph3_without_00 = np.copy(hist_ph3)
    # hist_ph1_without_00[0, 0] = 0
    # hist_ph2_without_00[0, 0] = 0
    # hist_ph3_without_00[0, 0] = 0
    #
    # # In[18]:
    # # DRAW: Draws the action heatmaps for low medium and high sofa levels for RL actions.
    # deeprl2_actions_low_iv = deeprl2_actions_low_tuple[:, 0]
    # deeprl2_actions_low_vaso = deeprl2_actions_low_tuple[:, 1]
    # hist_drl1, _, _ = np.histogram2d(deeprl2_actions_low_iv, deeprl2_actions_low_vaso, bins=5)
    #
    # deeprl2_actions_mid_iv = deeprl2_actions_mid_tuple[:, 0]
    # deeprl2_actions_mid_vaso = deeprl2_actions_mid_tuple[:, 1]
    # hist_drl2, _, _ = np.histogram2d(deeprl2_actions_mid_iv, deeprl2_actions_mid_vaso, bins=5)
    #
    # deeprl2_actions_high_iv = deeprl2_actions_high_tuple[:, 0]
    # deeprl2_actions_high_vaso = deeprl2_actions_high_tuple[:, 1]
    # hist_drl3, _, _ = np.histogram2d(deeprl2_actions_high_iv, deeprl2_actions_high_vaso, bins=5)
    #
    # # Create plots without the domination (0,0) cell to make the rest more visible
    # hist_drl1_without_00 = np.copy(hist_drl1)
    # hist_drl2_without_00 = np.copy(hist_drl2)
    # hist_drl3_without_00 = np.copy(hist_drl3)
    # hist_drl1_without_00[0, 0] = 0
    # hist_drl2_without_00[0, 0] = 0
    # hist_drl3_without_00[0, 0] = 0
    # # In[19]:
    #
    # x_edges = np.arange(-0.5, 5)
    # y_edges = np.arange(-0.5, 5)
    #
    # # In[20]:
    #
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    # ax1.imshow(np.flipud(hist_drl1), cmap="Blues", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # ax2.imshow(np.flipud(hist_drl2), cmap="OrRd", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # ax3.imshow(np.flipud(hist_drl3), cmap="Greens", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    #
    # # ax1.grid(color='b', linestyle='-', linewidth=1)
    # # ax2.grid(color='r', linestyle='-', linewidth=1)
    # # ax3.grid(color='g', linestyle='-', linewidth=1)
    #
    # # Major ticks
    # ax1.set_xticks(np.arange(0, 5, 1))
    # ax1.set_yticks(np.arange(0, 5, 1))
    # ax2.set_xticks(np.arange(0, 5, 1))
    # ax2.set_yticks(np.arange(0, 5, 1))
    # ax3.set_xticks(np.arange(0, 5, 1))
    # ax3.set_yticks(np.arange(0, 5, 1))
    #
    # # Labels for major ticks
    # ax1.set_xticklabels(np.arange(0, 5, 1))
    # ax1.set_yticklabels(np.arange(0, 5, 1))
    # ax2.set_xticklabels(np.arange(0, 5, 1))
    # ax2.set_yticklabels(np.arange(0, 5, 1))
    # ax3.set_xticklabels(np.arange(0, 5, 1))
    # ax3.set_yticklabels(np.arange(0, 5, 1))
    #
    # # Minor ticks
    # ax1.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax1.set_yticks(np.arange(-.5, 5, 1), minor=True)
    # ax2.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax2.set_yticks(np.arange(-.5, 5, 1), minor=True)
    # ax3.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax3.set_yticks(np.arange(-.5, 5, 1), minor=True)
    #
    # # Gridlines based on minor ticks
    # ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    # ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
    # ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)
    #
    # im1 = ax1.pcolormesh(x_edges, y_edges, hist_drl1, cmap='Blues')
    # f.colorbar(im1, ax=ax1)
    # #     f.colorbar(im1, ax=ax1, label = "Action counts")
    #
    # im2 = ax2.pcolormesh(x_edges, y_edges, hist_drl2, cmap='Greens')
    # f.colorbar(im2, ax=ax2)
    # #     f.colorbar(im2, ax=ax2, label = "Action counts")
    #
    # im3 = ax3.pcolormesh(x_edges, y_edges, hist_drl3, cmap='OrRd')
    # f.colorbar(im3, ax=ax3)
    # #     f.colorbar(im3, ax=ax3, label = "Action counts")
    #
    # big_size = 20
    # BIGGER_SIZE = 20
    # ax1.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax2.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax3.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax1.set_xlabel('Vasopressor dose', fontsize=big_size)
    # ax2.set_xlabel('Vasopressor dose', fontsize=big_size)
    # ax3.set_xlabel('Vasopressor dose', fontsize=big_size)
    #
    # ax1.set_title("Low SOFA policy - Agent", fontsize=big_size)
    # ax2.set_title("Mid SOFA policy - Agent", fontsize=big_size)
    # ax3.set_title("High SOFA policy - Agent", fontsize=big_size)
    #
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax2.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax3.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax3.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # plt.tight_layout()
    # plt.savefig(res_dir + 'iter_' + iteration_idx + '_RL_Action_Heatmap.png', dpi=300)
    #
    # # In[21]:
    #
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    # ax1.imshow(np.flipud(hist_ph1), cmap="Blues", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # ax2.imshow(np.flipud(hist_ph2), cmap="OrRd", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # ax3.imshow(np.flipud(hist_ph3), cmap="Greens", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    #
    # # ax1.grid(color='b', linestyle='-', linewidth=1)
    # # ax2.grid(color='r', linestyle='-', linewidth=1)
    # # ax3.grid(color='g', linestyle='-', linewidth=1)
    #
    # # Major ticks
    # ax1.set_xticks(np.arange(0, 5, 1))
    # ax1.set_yticks(np.arange(0, 5, 1))
    # ax2.set_xticks(np.arange(0, 5, 1))
    # ax2.set_yticks(np.arange(0, 5, 1))
    # ax3.set_xticks(np.arange(0, 5, 1))
    # ax3.set_yticks(np.arange(0, 5, 1))
    #
    # # Labels for major ticks
    # ax1.set_xticklabels(np.arange(0, 5, 1))
    # ax1.set_yticklabels(np.arange(0, 5, 1))
    # ax2.set_xticklabels(np.arange(0, 5, 1))
    # ax2.set_yticklabels(np.arange(0, 5, 1))
    # ax3.set_xticklabels(np.arange(0, 5, 1))
    # ax3.set_yticklabels(np.arange(0, 5, 1))
    #
    # # Minor ticks
    # ax1.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax1.set_yticks(np.arange(-.5, 5, 1), minor=True)
    # ax2.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax2.set_yticks(np.arange(-.5, 5, 1), minor=True)
    # ax3.set_xticks(np.arange(-.5, 5, 1), minor=True)
    # ax3.set_yticks(np.arange(-.5, 5, 1), minor=True)
    #
    # # Gridlines based on minor ticks
    # ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    # ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
    # ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)
    #
    # im1 = ax1.pcolormesh(x_edges, y_edges, hist_ph1, cmap='Blues')
    # f.colorbar(im1, ax=ax1)
    #
    # im2 = ax2.pcolormesh(x_edges, y_edges, hist_ph2, cmap='Greens')
    # f.colorbar(im2, ax=ax2)
    #
    # im3 = ax3.pcolormesh(x_edges, y_edges, hist_ph3, cmap='OrRd')
    # f.colorbar(im3, ax=ax3)
    #
    # big_size = 20
    # BIGGER_SIZE = 20
    # ax1.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax2.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax3.set_ylabel('IV fluid dose', fontsize=big_size)
    # ax1.set_xlabel('Vasopressor dose', fontsize=big_size)
    # ax2.set_xlabel('Vasopressor dose', fontsize=big_size)
    # ax3.set_xlabel('Vasopressor dose', fontsize=big_size)
    #
    # ax1.set_title("Low SOFA policy - Clinician", fontsize=big_size)
    # ax2.set_title("Mid SOFA policy - Clinician", fontsize=big_size)
    # ax3.set_title("High SOFA policy - Clinician", fontsize=big_size)
    #
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax2.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax3.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    # for tick in ax3.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(BIGGER_SIZE)
    #
    # plt.tight_layout()
    # plt.savefig(res_dir + 'iter_' + iteration_idx + '_Phys_Action_Heatmap.png', dpi=300)

    def make_iv_plot_data(df_diff):
        bin_medians_iv = []
        mort_iv = []
        mort_std_iv = []
        i = -800
        while i <= 900:
            count = df_diff.loc[(df_diff['iv_diff'] > i - 50) & (df_diff['iv_diff'] < i + 50)]
            try:
                res = sum(count['mort']) / float(len(count))
                if len(count) >= 2:
                    bin_medians_iv.append(i)
                    mort_iv.append(res)
                    mort_std_iv.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += 100
        return bin_medians_iv, mort_iv, mort_std_iv

    # In[24]:

    def make_vaso_plot_data(df_diff):
        bin_medians_vaso = []
        mort_vaso = []
        mort_std_vaso = []
        i = -0.6
        while i <= 0.6:
            count = df_diff.loc[(df_diff['vaso_diff'] > i - 0.05) & (df_diff['vaso_diff'] < i + 0.05)]
            try:
                res = sum(count['mort']) / float(len(count))
                if len(count) >= 2:
                    bin_medians_vaso.append(i)
                    mort_vaso.append(res)
                    mort_std_vaso.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += 0.1
        return bin_medians_vaso, mort_vaso, mort_std_vaso

    # In[25]:

    df_diff_low = make_df_diff(deeprl2_actions_low, df_test_orig_low)
    df_diff_mid = make_df_diff(deeprl2_actions_mid, df_test_orig_mid)
    df_diff_high = make_df_diff(deeprl2_actions_high, df_test_orig_high)

    # In[26]:

    bin_med_iv_deep_low, mort_iv_deep_low, mort_std_iv_deep_low = make_iv_plot_data(df_diff_low)
    bin_med_vaso_deep_low, mort_vaso_deep_low, mort_std_vaso_deep_low = make_vaso_plot_data(df_diff_low)

    bin_med_iv_deep_mid, mort_iv_deep_mid, mort_std_iv_deep_mid = make_iv_plot_data(df_diff_mid)
    bin_med_vaso_deep_mid, mort_vaso_deep_mid, mort_std_vaso_deep_mid = make_vaso_plot_data(df_diff_mid)

    bin_med_iv_deep_high, mort_iv_deep_high, mort_std_iv_deep_high = make_iv_plot_data(df_diff_high)
    bin_med_vaso_deep_high, mort_vaso_deep_high, mort_std_vaso_deep_high = make_vaso_plot_data(df_diff_high)

    def diff_plot(med_vaso, mort_vaso, std_vaso, med_iv, mort_iv, std_iv, col, title):
        big_size = 20
        BIGGER_SIZE = 14
        f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 4))
        step = 2
        if col == 'r':
            fillcol = 'lightsalmon'
        elif col == 'g':
            fillcol = 'palegreen'
            step = 1
        elif col == 'b':
            fillcol = 'lightblue'
        ax1.plot(med_vaso, sliding_mean(mort_vaso), color=col)
        ax1.fill_between(med_vaso, sliding_mean(mort_vaso) - 1 * std_vaso,
                         sliding_mean(mort_vaso) + 1 * std_vaso, color=fillcol)
        t = title + ": Vasopressors"
        ax1.set_title(t, fontsize=big_size)
        #     ax1.set_xlabel(t)
        x_r = [i / 10.0 for i in range(-6, 8, 2)]

        y_r = [i / 20.0 for i in range(0, 20, step)]
        ax1.set_xticks(x_r)
        ax1.set_yticks(y_r)
        ax1.grid()

        ax2.plot(med_iv, sliding_mean(mort_iv), color=col)
        ax2.fill_between(med_iv, sliding_mean(mort_iv) - 1 * std_iv,
                         sliding_mean(mort_iv) + 1 * std_iv, color=fillcol)
        t = title + ": IV fluids"
        ax2.set_title(t, fontsize=big_size)
        x_iv = [i for i in range(-800, 900, 400)]
        ax2.set_xticks(x_iv)
        ax2.grid()

        ax1.set_ylabel('Observed Mortality', rotation='vertical', fontsize=big_size)

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(BIGGER_SIZE)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(BIGGER_SIZE)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(BIGGER_SIZE)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(BIGGER_SIZE)
        f.savefig(res_dir + 'iter_' + iteration_idx + '_' + title + '_Action_vs_mortality.png', dpi=300)

    # In[38]:

    #     def diff_plot(med_vaso, mort_vaso, std_vaso, med_iv, mort_iv, std_iv, col, title):
    #         f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row', figsize = (10,4))
    #         step = 2
    #         if col == 'r':
    #             fillcol = 'lightsalmon'
    #         elif col == 'g':
    #             fillcol = 'palegreen'
    #             step = 1
    #         elif col == 'b':
    #             fillcol = 'lightblue'
    #         ax1.plot(med_vaso, sliding_mean(mort_vaso), color=col)
    #         ax1.fill_between(med_vaso, sliding_mean(mort_vaso) - 1*std_vaso,
    #                          sliding_mean(mort_vaso) + 1*std_vaso, color=fillcol)
    #         t = title + ": Vasopressors"
    #         ax1.set_title(t)
    #         x_r = [i/10.0 for i in range(-6,8,2)]

    #         y_r = [i/20.0 for i in range(0,20,step)]
    #         ax1.set_xticks(x_r)
    #         ax1.set_yticks(y_r)
    #         ax1.grid()

    #         ax2.plot(med_iv, sliding_mean(mort_iv), color=col)
    #         ax2.fill_between(med_iv, sliding_mean(mort_iv) - 1*std_iv,
    #                          sliding_mean(mort_iv) + 1*std_iv, color=fillcol)
    #         t = title + ": IV fluids"
    #         ax2.set_title(t)
    #         x_iv = [i for i in range(-800,900,400)]
    #         ax2.set_xticks(x_iv)
    #         ax2.grid()

    #         f.text(0.3, -0.03, 'Difference between optimal and physician vasopressor dose', ha='center', fontsize=10)
    #         f.text(0.725, -0.03, 'Difference between optimal and physician IV dose', ha='center', fontsize=10)
    #         f.text(0.05, 0.5, 'Observed Mortality', va='center', rotation='vertical', fontsize = 10)
    #         f.savefig(res_dir +title+'_Action_vs_mortality.png',dpi = 300)

    # In[39]:

    diff_plot(bin_med_vaso_deep_low, mort_vaso_deep_low, mort_std_vaso_deep_low,
              bin_med_iv_deep_low, mort_iv_deep_low, mort_std_iv_deep_low, 'b', 'Low SOFA')

    # In[29]:

    diff_plot(bin_med_vaso_deep_mid, mort_vaso_deep_mid, mort_std_vaso_deep_mid,
              bin_med_iv_deep_mid, mort_iv_deep_mid, mort_std_iv_deep_mid, 'g', 'Medium SOFA')

    # In[30]:

    diff_plot(bin_med_vaso_deep_high, mort_vaso_deep_high, mort_std_vaso_deep_high,
              bin_med_iv_deep_high, mort_iv_deep_high, mort_std_iv_deep_high, 'r', 'High SOFA')

    # draw the q_vs_mortality plot
    data['phys_action'] = data['iv_fluids_quantile'] * 5 + data['vasopressors_quantile'] - 6
    q_vals_phys = data.apply(lambda x: x['Q_' + str(int(x[phys_action]))], axis=1)
    pp = pd.Series(q_vals_phys)
    phys_df = pd.DataFrame(pp)
    phys_df['mort'] = copy.deepcopy(np.array(data['mortality_hospital']))

    bin_medians = []
    mort = []
    mort_std = []
    k = (np.max(q_vals_phys) - np.min(q_vals_phys)) / 50
    i = q_vals_phys.quantile([0.01, 0.99]).values[0]
    while (i <= q_vals_phys.quantile([0.01, 0.99]).values[1]):
        count = phys_df.loc[(phys_df[0] > i - k) & (phys_df[0] < i + k)]
        try:
            res = sum(count['mort']) / float(len(count))
            if len(count) >= 2:
                bin_medians.append(i)
                mort.append(res)
                mort_std.append(sem(count['mort']))
        except ZeroDivisionError:
            pass
        i += (q_vals_phys.quantile([0.01, 0.99]).values[1] - q_vals_phys.quantile([0.01, 0.99]).values[0]) / 30

    def sliding_mean(data_array, window=2):
        new_list = []
        for i in range(len(data_array)):
            indices = range(max(i - window + 1, 0),
                            min(i + window + 1, len(data_array)))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)
        return np.array(new_list)

    plt.figure(figsize=(6, 5))
    plt.plot(bin_medians, sliding_mean(mort))
    plt.fill_between(bin_medians, sliding_mean(mort) - 1 * sliding_mean(mort_std),
                     sliding_mean(mort) + 1 * sliding_mean(mort_std), color='#ADD8E6')
    plt.grid()
    # plt.yticks(range(0.0,0.5,5))
    r = [float(i) / 10 for i in range(1, 11, 1)]
    _ = plt.yticks(r, fontsize=12)
    # _ = plt.title("Mortality vs Expected Return", fontsize=15)
    _ = plt.ylabel("Proportion Mortality", fontsize=20)
    _ = plt.xlabel("Expected Return", fontsize=18)
    plt.xticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(res_dir + 'iter_' + iteration_idx + '_q_vs_mortality.png', dpi=300)
    plt.close('all')

    #     REWARD_FUN = setting.REWARD_FUN
    #     data['reward']= data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    #     data['reward']= data['reward_mortality']
    #     optimality_score_columns = ['done', 'ori_mbp', 'next_ori_mbp', 'ori_lactate', 'next_ori_lactate', 'ori_sofa_24hours', 'next_ori_sofa_24hours']
    # data['reward'] = data['rewards_r1_or_r2']
    # data.drop(columns=['rewards_r1_or_r2'])
    # data['reward'] = data[optimality_score_columns].apply(lambda row: utils.score_mortality_sofa_lactate_mbp_v1(row), axis=1)

    # Quantitavie evaluation of the RL policy
    res_dt = q_vs_outcome(data)
    q_dr_dt = quantitive_eval(data, res_dt)
    conc_dt = action_concordant_rate(data)
    #     plot_loss(loss)

    print(q_dr_dt)
    print(conc_dt)
