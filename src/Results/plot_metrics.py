import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(experiments, split):
    """Load all metrics for all the runs in the experiments.

    Args:
        experiments (dict<str: Path>): An experiment display name and its path.
        split (str): The split to load the metrics from.
    
    Returns:
        dict<str, list<dict<str, float>>: Dictionary with an experiment name and a list of its metrics' dictionary for each run
    """
    metrics = defaultdict(list)
    for exp, exp_path in experiments.items():
        assert exp_path.exists(), f"[plot_metrics] Path {exp_path} does not exist."
        for run in [exp_path/run_name for run_name in exp_path.glob('*') if (exp_path/run_name).is_dir()]:
            with (run/"metric.json").open("r") as f:
                run_metric = json.load(f)
            metrics[exp].append(run_metric[split])
            
    return metrics

def plot_metric(ax, metrics, metric_name, exp_colors, y_label = '', title = ''):
    """Plot a boxplot and its mean value for a given metric and different experiments.

    Args:
        ax (plt.Axis): The ax to display the plot
        metrics dict<str, list<dict<str, float>>: Dictionary with an experiment name and a list of its metrics' dictionary for each run
        metric_name (str): The name of the metric to be displayed
        exp_colors (dict<str, color>): Colors associated to each experiment
        y_label (str, optional): Y-axis label. Defaults to ''.
        title (str, optional): Plot title. Defaults to ''.
    """
    exp_names = list(metrics.keys())

    bp = ax.boxplot([[metrics[e][i][metric_name] for i in range(len(metrics[e]))] for e in exp_names], 
                positions=np.arange(1, len(exp_names) + 1), widths=0.6, patch_artist=False, showmeans=True, meanprops = dict(marker='.', markeredgecolor='black', markerfacecolor='black'))
    for j, line in enumerate(bp['medians']):
        plt.setp(line, color=exp_colors[exp_names[j]], linestyle="-")
    for j, line in enumerate(bp['boxes']):
        plt.setp(line, color=exp_colors[exp_names[j]], linestyle="-")
    print(f"Means {metric_name}", {e: bp['means'][j].get_ydata() for j, e in enumerate(exp_names)})
    # Customizing the plot
    ax.set_xticks(np.arange(1, len(exp_names) + 1), labels = exp_names, rotation = 18, ha = "right")
    ax.set_xlabel('Experiments')
    ax.set_ylabel(' '.join(metric_name.split('_')) if y_label == '' else y_label)
    ax.set_title(title)
    ax.grid(True)
    legend_lines = [plt.Line2D([0], [0], color=exp_colors[exp_name], linestyle="-", lw=2) for exp_name in exp_names]
    ax.legend(legend_lines, exp_names)

def plot_metrics_by_exp(ax, metrics, metric_names, colors, y_label = '', title = '', legend_names = None):
    """Plot a boxplot and its mean value for several metrics and their associated colors and different experiments.

    Args:
        ax (plt.Axis): The ax to display the plot
        metrics dict<str, list<dict<str, float>>: Dictionary with an experiment name and a list of its metrics' dictionary for each run
        metric_names (str): The names of the metrics to be displayed
        colors (dict<str, color>): Colors associated to each metric
        y_label (str, optional): Y-axis label. Defaults to ''.
        title (str, optional): Plot title. Defaults to ''.
        legend_names (list<str>, optional): Legends for the metrics. Defaults to None (metrics raw names are displayed).
    """
    exp_names = list(metrics.keys())

    positions = []
    for e in range(len(exp_names)):
        pos = np.arange(1, len(metric_names) + 1) + e * (len(metric_names) + 1)
        positions.append(pos)
        bp = ax.boxplot([[metrics[exp_names[e]][i][metric_name] for i in range(len(metrics[exp_names[e]]))] for metric_name in metric_names], 
                    positions=pos, widths=0.6, patch_artist=False, showmeans=True, meanprops = dict(marker='.', markeredgecolor='black', markerfacecolor='black'))
        for j, line in enumerate(bp['medians']):
            plt.setp(line, color=colors[metric_names[j]], linestyle="-")
        for j, line in enumerate(bp['boxes']):
            plt.setp(line, color=colors[metric_names[j]], linestyle="-")
        print(f"Means {exp_names[e]}", {m: bp['means'][j].get_ydata() for j, m in enumerate(metric_names)})
    # Customizing the plot
    ax.set_xticks([np.mean(p) for p in positions], labels = exp_names, rotation = 18, ha = "right")
    ax.set_xlabel('Experiments')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    legend_lines = [plt.Line2D([0], [0], color=colors[m], linestyle="-", lw=2) for m in metric_names]
    ax.legend(legend_lines, metric_names if legend_names is None else legend_names)