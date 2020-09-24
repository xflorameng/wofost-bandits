from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sns


class ReadCSV:
    """Read simulation data into df."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        assert df.notnull().values.all(), 'Missing values'
        for col in {'expert_weights', 'expert_mean_rewards'}:
            df[col] = df[col].str.strip('[]').apply(lambda x: np.array(x.split()).astype(float))
        self.df = df

    def add_regret(self):
        assert 'best_expert_mean_reward' in self.df.columns, 'best_expert_mean_reward unrecorded'
        if 'regret' not in self.df.columns:
            self.df['regret'] = self.df['best_expert_mean_reward'] - self.df['reward']


def plot_time_series(csv_path, y, ylabel=None, cumsum=True,
                     style='whitegrid', save=False, filename_tag='', timestamp=False, plot_path=''):
    """Plot time series"""

    data = ReadCSV(csv_path)
    if y == 'regret':
        data.add_regret()
    data.df.set_index('time', inplace=True)
    s = data.df[y]
    if cumsum:
        s = s.cumsum()
    sns.set_theme(style=style)
    lineplot = sns.lineplot(data=s, palette="tab10", linewidth=2.5)
    lineplot.set_xlabel('Time')
    if ylabel is None:
        ylabel = f'Cumulative {y}'
    lineplot.set_ylabel(ylabel)
    if save:
        if filename_tag == '':
            filename_tag = y
        if timestamp:
            filename = f'time_series_{filename_tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        else:
            filename = f'time_series_{filename_tag}.pdf'
        plt.savefig(plot_path + filename, transparent=True)
    plt.show()
    plt.pause(3)
    plt.close()


def plot_bar_chart(csv_path, xticklabels=None, sort_by_reward=True,
                   style='whitegrid', save=False, filename_tag='expert_weight_reward', timestamp=False, plot_path=''):
    """Plot bar chart"""

    data = ReadCSV(csv_path)
    expert_final_weights = data.df.loc[data.df.index[-1], 'expert_weights']
    expert_mean_rewards = np.array(data.df['expert_mean_rewards'].values.tolist())
    expert_mean_rewards_over_time = np.sum(expert_mean_rewards, axis=0) / len(expert_mean_rewards)
    sns.set_theme(style=style)
    if xticklabels is None:
        xticklabels = np.arange(len(expert_final_weights))
        rotate_xticklabels = True
    else:
        rotate_xticklabels = False
    x = np.arange(len(expert_final_weights))
    if sort_by_reward:
        sorted_ids = expert_mean_rewards_over_time.argsort()
        expert_final_weights = expert_final_weights[sorted_ids]
        expert_mean_rewards_over_time = expert_mean_rewards_over_time[sorted_ids]
        xticklabels = np.array(xticklabels)[sorted_ids]
    width = .35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, expert_final_weights, width, label='Final weight')
    ax.bar(x + width / 2, expert_mean_rewards_over_time, width, label='Mean reward per time step')
    ax.set_xlabel('Expert')
    ax.set_xticks(x)
    if rotate_xticklabels:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    if save:
        if timestamp:
            filename = f'bar_chart_{filename_tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        else:
            filename = f'bar_chart_{filename_tag}.pdf'
        plt.savefig(plot_path + filename, transparent=True)
    plt.show()
    plt.pause(3)
    plt.close()


def animation_bar_chart(csv_path, xticklabels=None, ylim=None, sort_by_reward=True,
                        style='whitegrid', save=False, filename_tag='expert_weight', timestamp=False, plot_path=''):
    """Animate bar chart"""

    data = ReadCSV(csv_path)
    data.df.set_index('time', inplace=True)
    expert_weights = data.df['expert_weights']
    n_experts = len(expert_weights.loc[1])
    sns.set_theme(style=style)
    if xticklabels is None:
        xticklabels = np.arange(n_experts)
        rotate_xticklabels = True
    else:
        rotate_xticklabels = False
    x = np.arange(n_experts)
    if ylim is None:
        expert_final_weights = expert_weights.loc[len(expert_weights)]
        ylim = (0, min(1, 1.1*np.max(expert_final_weights)))
    if sort_by_reward:
        expert_mean_rewards = np.array(data.df['expert_mean_rewards'].values.tolist())
        expert_mean_rewards_over_time = np.sum(expert_mean_rewards, axis=0) / len(expert_mean_rewards)
        sorted_ids = expert_mean_rewards_over_time.argsort()
        for t in range(1, len(expert_weights)+1):
            expert_weights.at[t] = expert_weights.at[t][sorted_ids]
        xticklabels = np.array(xticklabels)[sorted_ids]
    width = .6
    fig, ax = plt.subplots()

    def init():
        ax.clear()
        ax.set_xlim(-.5, n_experts-.5)
        ax.set_ylim(*ylim)

    def animate(frame_id):
        init()
        time = frame_id + 1
        ax.bar(x, expert_weights.loc[time], width)
        ax.set_xlabel('Expert')
        ax.set_xticks(x)
        ax.set_ylabel('Weight')
        ax.set_title(f'Time {time}')
        if rotate_xticklabels:
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticklabels(xticklabels, rotation=45, ha='right')
        fig.tight_layout()

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(expert_weights), interval=200, repeat=False)

    if save:
        if timestamp:
            filename = f'animated_bar_chart_{filename_tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        else:
            filename = f'animated_bar_chart_{filename_tag}.mp4'
        FFwriter = FFMpegWriter(fps=10)
        anim.save(plot_path + filename, writer=FFwriter)
    plt.show()
    plt.pause(3)
    plt.close()
