import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import font_manager
import matplotlib
from matplotlib.patches import Patch
import argparse
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from tools import calc_pvalue

script_dir = Path(__file__).resolve().parent
font_path = script_dir / 'Arial.ttf'

matplotlib.rcParams["font.family"] = "Arial"
arial_font = font_manager.FontProperties(fname=font_path, size=6)
color_models = {'ST-GCN (Ours)': '#B55358', 'CTR-GCN': '#2E78B0', 'PoseFormerV2': '#756BB1', 'SkateFormer': '#699583', 'DSTformer': '#767171'}
color_sub_groups = {
    'ST-GCN (Ours)': {'Baseline': '#DFB3B5', 'Baseline+Causality': '#D79397', 'Baseline+Pretraining': '#C87378', 'Final Model': '#B55358'},
    'CTR-GCN': {'Baseline': '#B0D1EA', 'Baseline+Causality': '#85B3D7', 'Baseline+Pretraining': '#5996C3', 'Final Model': '#2E78B0'},
    'PoseFormerV2': {'Baseline': '#D0CDE5', 'Baseline+Causality': '#B1ADD4', 'Baseline+Pretraining': '#938CC3', 'Final Model': '#756BB1'},
    'SkateFormer': {'Baseline': '#B3C9C0', 'Baseline+Causality': '#9EB9AE', 'Baseline+Pretraining': '#82A89A', 'Final Model': '#699583'},
    'DSTformer': {'Baseline': '#C0BCBC', 'Baseline+Causality': '#ADA9A9', 'Baseline+Pretraining': '#8F8A8A', 'Final Model': '#767171'}}


def plot_bars(plot_data, color_dict, figsize=(6, 6), save_path=None, bar_width=0.15, xlim=None):
    n_groups = len(plot_data)
    group_idx = np.arange(n_groups)
    n_bars = len(color_dict)
    bar_w = bar_width
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w * 1.25
    x_ticks_positions = []
    x_ticks_names = []
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for g, group in enumerate(plot_data):
        for j, bar_name in enumerate(plot_data[group]):
            ax.bar(group_idx[g] + offsets[j], plot_data[group][bar_name]['mean'], width=bar_w, color=color_dict[group][bar_name], alpha=0.9)
            x_ticks_positions.append(group_idx[g] + offsets[j])
            if bar_name in map_name:
                x_ticks_names.append(map_name[bar_name])
            else:
                x_ticks_names.append(bar_name)
        pval_pair = (('Baseline', 'Final Model'), ('Baseline+Causality', 'Final Model'), ('Baseline+Pretraining', 'Final Model'))
        pval_dict = {}
        for idx in range(len(pval_pair)):
            pair = pval_pair[idx]
            pval = calc_pvalue(plot_data[group][pair[1]]['values'], plot_data[group][pair[0]]['values'])
            pval_dict[pair] = pval
    if xlim:
        plt.xlim(xlim)
    plt.ylim([75, 95])
    ticks = [75, 80, 85, 90, 95]
    plt.yticks(ticks)
    plt.yticks(fontproperties=arial_font)
    plt.xticks(x_ticks_positions, x_ticks_names, ha='right', rotation=35, fontproperties=arial_font)
    plt.ylabel('AUROC', fontproperties=arial_font)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_time_bars(plot_data, color_dict, figsize, ylabel, save_path, ylim=None, yticks=None, bar_width=0.15):
    plt.figure(figsize=figsize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    n_bars = len(plot_data)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width * 1.2
    for index, model in enumerate(color_dict):
        plt.bar(offsets[index], plot_data[model], color=color_dict[model], width=bar_width, alpha=0.85)
    plt.ylabel(ylabel, fontproperties=arial_font)
    if ylim:
        plt.ylim(ylim)
        if yticks:
            plt.yticks(yticks)
    plt.yticks(fontproperties=arial_font)
    plt.xticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_bar_legend(color_dict, figsize, save_path):
    handles = [Patch(facecolor=color, label=label) for label, color in color_dict.items()]
    plt.figure(figsize=figsize)
    plt.legend(handles=handles, loc='center', ncol=len(color_dict), frameon=False, prop=arial_font, handlelength=1.5, handletextpad=0.5, columnspacing=0.7)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_statistics(data_list):
    mean_val = np.mean(data_list)
    return {'mean': mean_val, 'values': data_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Extended Data Figure 4')
    parser.add_argument('--file-path', type=str, default='./results_examples/results_backbone_comparison.xlsx',
                        help='Path to the Excel file with results')
    parser.add_argument('--save-dir', type=str, default='./figures/examples/Extended_Data_Fig4',
                        help='Directory to save output figures')
    args = parser.parse_args()
    file_path = args.file_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(file_path, sheet_name="Sheet1")
    map_name = {'Final Model': 'Final model'}
    file_columns = [col for col in df.columns if col not in ['dataset']]
    exp_list = ['Baseline', 'Baseline+Causality', 'Baseline+Pretraining', 'Final Model']
    results = {}
    for m in color_models:
        results[m] = {}
    for set in file_columns:
        model = set.split('_')[0]
        setting = set.split('_')[1]
        values = df[set].tolist()
        results[model].update({setting: calculate_statistics(values)})
    plot_bars(results, color_sub_groups, (4.5, 2), f'{save_dir}/a_backbone_replace.svg', 0.16, (-0.55, 4.35))
    plot_bar_legend(color_models, (3.8, 0.18), f'{save_dir}/legend.svg')
    df_time = pd.read_excel(file_path, sheet_name="Sheet2")
    df_time = df_time.set_index('model')
    file_columns = [col for col in df_time.columns if col != 'model']
    result_time = {'Pretraining time per epoch (s)': {}, 'GPU count for pretraining': {}, 'Inference time (ms)': {}}
    for col in file_columns:
        for model in color_models:
            result_time[col].update({model: df_time.loc[model, col]})
    time_fig_size = (1.5, 1.4)
    plot_time_bars(result_time['Pretraining time per epoch (s)'], color_models, time_fig_size,
                   'Pretraining time per epoch (s)', f'{save_dir}/b_pretrain_time.svg', [0, 160], [0, 40, 80, 120, 160], bar_width=0.07)
    plot_time_bars(result_time['GPU count for pretraining'], color_models, time_fig_size,
                   'GPU count for pretraining', f'{save_dir}/c_gpu_count.svg', [0, 8], [0, 4, 8], bar_width=0.07)
    plot_time_bars(result_time['Inference time (ms)'], color_models, time_fig_size,
                   'Inference time (ms)', f'{save_dir}/d_inference_time.svg', [0, 20], [0, 10, 20], bar_width=0.07)
    print(f'Extended Data Fig 4 -- {save_dir}')
