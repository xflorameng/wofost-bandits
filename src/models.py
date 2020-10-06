import os.path
import math
from math import exp, log, sqrt
import csv
import warnings
from datetime import datetime
import random
import time
import numpy as np

from pcse.fileinput import CABOFileReader
from pcse.db import NASAPowerWeatherDataProvider
from pcse.base import ParameterProvider
from pcse.engine import Engine

from src.actions import AgroActions


class UniformExpert:
    def __init__(self, actions):
        self.actions = actions
        self.n_actions = len(self.actions)

    def give_advice(self):
        return np.ones(self.n_actions) / self.n_actions

    def update(self, args):
        pass


class Expert:
    def __init__(self, actions):
        self.actions = actions
        self.n_actions = len(self.actions)
        self.stubborn_action = np.random.default_rng().choice(self.n_actions, shuffle=False)

    def give_advice(self):
        advice = np.zeros(self.n_actions)
        advice[self.stubborn_action] = 1
        return advice

    def update(self, args):
        pass


class NoActionExpert(Expert):
    def __init__(self, actions):
        super().__init__(actions)
        self.stubborn_action = 0


class WofostExpert(Expert):
    def __init__(self, actions):
        super().__init__(actions)
        self.yields = np.zeros(self.n_actions)

    def update(self, yields):
        self.yields = yields

    def give_advice(self):
        advice = self.yields / np.sum(self.yields)
        return advice


class PerturbedWofostExpert(WofostExpert):
    def __init__(self, actions, noise_scale=.1):
        super().__init__(actions)
        if not isinstance(noise_scale, float):
            raise TypeError('noise_scale must be float')
        if not noise_scale >= 0:
            raise ValueError('noise_scale must be nonnegative')
        self.noise_scale = noise_scale

    def give_advice(self):
        original_advice = self.yields / np.sum(self.yields)
        advice = np.zeros(self.n_actions)
        advice_sum = 0
        n_sampling = 0
        max_n_sampling = 5
        while advice_sum < 1e-3:
            if n_sampling < max_n_sampling:
                n_sampling += 1
                advice = original_advice + np.random.default_rng().normal(0, self.noise_scale, self.n_actions)
                advice = np.clip(advice, 0, 1e3)
                advice_sum = np.sum(advice)
            else:
                advice = np.ones(self.n_actions)
        advice = advice / np.sum(advice)
        return advice


class ComplementaryWofostExpert(WofostExpert):
    def __init__(self, actions):
        super().__init__(actions)

    def give_advice(self):
        advice = 1 - self.yields / np.sum(self.yields)
        advice = advice / np.sum(advice)
        return advice


class MinWofostExpert(WofostExpert):
    def __init__(self, actions):
        super().__init__(actions)

    def give_advice(self):
        idx = np.argmin(self.yields)
        advice = np.zeros(self.n_actions)
        advice[idx] = 1
        return advice


class MaxWofostExpert(WofostExpert):
    def __init__(self, actions):
        super().__init__(actions)

    def give_advice(self):
        idx = np.argmax(self.yields)
        advice = np.zeros(self.n_actions)
        advice[idx] = 1
        return advice


class PerturbedMaxWofostExpert(WofostExpert):
    def __init__(self, actions, noise_scale=.1):
        super().__init__(actions)
        if not isinstance(noise_scale, float):
            raise TypeError('noise_scale must be float')
        if not noise_scale >= 0:
            raise ValueError('noise_scale must be nonnegative')
        self.noise_scale = noise_scale

    def give_advice(self):
        idx = np.argmax(self.yields)
        original_advice = np.zeros(self.n_actions)
        original_advice[idx] = 1
        advice = np.zeros(self.n_actions)
        advice_sum = 0
        n_sampling = 0
        max_n_sampling = 5
        while advice_sum < 1e-3:
            if n_sampling < max_n_sampling:
                n_sampling += 1
                advice = original_advice + np.random.default_rng().normal(0, self.noise_scale, self.n_actions)
                advice = np.clip(advice, 0, 1e3)
                advice_sum = np.sum(advice)
            else:
                advice = np.ones(self.n_actions)
        advice = advice / np.sum(advice)
        return advice


class Wofost:
    @staticmethod
    def init_wofost():
        data_dir = os.path.join(os.getcwd(), '../simulation/default_data')
        crop_file_name = 'crop.cab'
        soil_file_name = 'soil.cab'
        site_file_name = 'site.cab'
        config_file_name = 'WLP_NPK.conf'

        soil_data = CABOFileReader(os.path.join(data_dir, soil_file_name))
        site_data = CABOFileReader(os.path.join(data_dir, site_file_name))
        crop_data = CABOFileReader(os.path.join(data_dir, crop_file_name))
        config = os.path.join(data_dir, config_file_name)

        params = ParameterProvider(crop_data, site_data, soil_data)
        latitude, longitude = 51.97, 5.67  # Wageningen, Netherlands
        wdp = NASAPowerWeatherDataProvider(latitude, longitude)

        return params, wdp, config

    @staticmethod
    def run_wofost(agromanagement, params, wdp, config):
        wofost = Engine(params, wdp, agromanagement, config)  # WLP_NPK
        wofost.run_till_terminate()
        r = wofost.get_summary_output()
        return r[0]['TWSO']  # Can be changed according to crop choice


class Exp4R:
    def __init__(self, actions, experts, T, delta=.1, rho=None):
        if not isinstance(actions, (list, tuple)):
            raise TypeError('actions must be either list or tuple')
        if not len(actions) > 0:
            raise ValueError('actions cannot be empty')
        self.actions = actions
        self.n_actions = len(self.actions)

        if not isinstance(experts, (list, tuple)):
            raise TypeError('experts must be either list or tuple')
        if not len(experts) > 0:
            raise ValueError('experts cannot be empty')
        self.experts = experts
        self.n_experts = len(experts)

        if not isinstance(T, int):
            raise TypeError('T must be int')
        if not T > 0:
            raise ValueError('T must be positive')
        self.T = T

        if not isinstance(delta, float):
            raise TypeError('delta must be float')
        if not 0 < delta <= 1:
            raise ValueError('delta must be positive and no larger than 1')
        self.delta = delta

        self.beta = sqrt(log(2 * self.n_experts / self.delta) / self.n_actions / self.T)
        assert self.beta ** 2 <= math.e - 2, 'ln(2N/delta) <= (e-2)KT must hold'

        if rho is None:
            rho = sqrt(log(self.n_experts) / self.n_actions / self.T)
        else:
            warnings.warn('For a customized value of rho, theoretical results may not hold.')
            if not isinstance(rho, float):
                raise TypeError('rho must be float')
        if not 0 < rho <= 1 / self.n_actions:
            raise ValueError(f'rho must be positive and no larger than {1 / self.n_actions}')
        else:
            self.rho = rho

        if not 49 * self.n_actions * log(2 * self.n_experts / self.delta) < self.T:
            warnings.warn('The regret upper bound is vacuous for the inputs given.\n'
                          'Try one or multiple of the following options:\n'
                          '- increasing time horizon\n'
                          '- increasing delta\n'
                          '- reducing the number of actions\n'
                          '- reducing the number of experts')

        self.t = 1

        self.expert_weights = np.ones(self.n_experts)
        self.expert_thresholds = np.zeros(self.n_experts)

        self.advice = np.empty((self.n_experts, self.n_actions))
        self.action_pmf = np.empty(self.n_actions)

    def get_advice(self):
        for expert_id in range(self.n_experts):
            self.advice[expert_id] = self.experts[expert_id].give_advice()
            assert abs(1 - np.sum(self.advice[expert_id])) <= 1e-2, \
                f'Improper advice, expert_id: {expert_id}\nadvice:{self.advice[expert_id]}'

    def combine_advice(self):
        self.action_pmf = ((1 - self.n_actions * self.rho) * np.matmul(self.expert_weights, self.advice)
                           / np.sum(self.expert_weights) + self.rho)

    def sample_action_id(self):
        return np.random.default_rng().choice(self.n_actions, p=self.action_pmf, shuffle=False)

    def update_weights(self, action_id, reward):
        mean_coeff = reward / self.action_pmf[action_id]
        for expert_id in range(self.n_experts):
            uncertainty = np.sum(np.divide(self.advice[expert_id], self.action_pmf))
            self.expert_weights[expert_id] *= exp(self.rho / 2 * (mean_coeff * self.advice[expert_id][action_id]
                                                                  + self.beta * uncertainty))
            self.expert_thresholds[expert_id] += uncertainty
        self.t += 1

    def get_expert_thresholds(self):
        self.expert_thresholds = ((1 + self.expert_thresholds / self.n_actions / self.T)
                                  * log(2 * self.n_experts / self.delta))

    def threshold_test(self):
        self.get_expert_thresholds()
        expert_weight_ranking = np.flip(self.expert_weights.argsort())
        log_expert_weights = np.log(self.expert_weights)
        pairwise_ranking = []
        for i in range(self.n_experts-1):
            better_id = expert_weight_ranking[i]
            j = self.n_experts - 1
            while j > i:
                worse_id = expert_weight_ranking[j]
                diff = log_expert_weights[better_id] - log_expert_weights[worse_id]
                if diff > self.expert_thresholds[better_id]:
                    pairwise_ranking.append((better_id, worse_id))
                else:
                    break
                j -= 1
        if len(pairwise_ranking) > 0:
            message = 'Estimated pairwise expert ranking as follows:'
            for better_id, worse_id in pairwise_ranking:
                message += f'\nExpert {better_id} better than Expert {worse_id}'
        else:
            message = 'Cannot decide on pairwise expert ranking...'
        print(message)

    def print_regret_upper_bound(self):
        regret_upper_bound = 7 * sqrt(self.n_actions * self.T * log(2 * self.n_experts / self.delta))
        print(f'Regret upper bound: {regret_upper_bound}')


class Environment:
    """Generate rewards and record history."""

    def __init__(self, actions, experts, record_best_expert=False, **kwargs):
        self.actions = actions
        self.n_actions = len(self.actions)
        self.experts = experts

        # WOFOST-related
        params, wdp, config = Wofost.init_wofost()
        self.wofost_params = [params, wdp, config]
        self.yields = np.empty(self.n_actions)

        self.current_rewards = np.empty(self.n_actions)
        self.record_best_expert = record_best_expert
        if 'means' in kwargs:
            assert len(kwargs['means']) == self.n_actions, 'means and actions must be the same length'
            self.means = kwargs['means']
        if 'alphas' in kwargs:
            assert len(kwargs['alphas']) == self.n_actions, 'alphas and actions must be the same length'
            self.alphas = kwargs['alphas']
        if 'betas' in kwargs:
            assert len(kwargs['betas']) == self.n_actions, 'betas and actions must be the same length'
            self.betas = kwargs['betas']
        self.t = 1
        self.history = {}

    @staticmethod
    def sample_random_year():
        """Sample year from a non-stationary distribution."""
        random.seed(time.time())
        years_complete_weather = list(range(1984, 2000)) + [2002] + list(range(2004, 2016)) + [2017, 2019]
        year = random.choice(years_complete_weather)
        return year

    def update(self):
        year = self.sample_random_year()
        self.actions, _ = AgroActions().create_actions([0, 1, 4, 7], [0, 15], year=year)
        # Run WOFOST to obtain yield for each action
        self.yields = np.zeros(self.n_actions)
        for i, action in enumerate(self.actions):
            self.yields[i] = Wofost.run_wofost(action, *self.wofost_params)
            assert self.yields[i] >= 0, 'Yield cannot be negative'
        # Provide yield information to experts in case they need it
        for expert in self.experts:
            expert.update(self.yields)

    def reward_bernoulli(self, action_id):
        self.current_rewards = (np.random.uniform(size=self.n_actions) < self.means) * 1
        return self.current_rewards[action_id]

    def reward_beta(self, action_id):
        self.current_rewards = np.random.default_rng().beta(self.alphas, self.betas)
        return self.current_rewards[action_id]

    def reward_wofost(self, action_id):
        self.current_rewards = self.yields / np.max(self.yields)
        return self.current_rewards[action_id]

    def add_history(self, action_id, reward, expert_weights, advice):
        expert_weights = expert_weights / np.sum(expert_weights)
        expert_mean_rewards = np.matmul(advice, self.current_rewards)
        self.history[self.t] = {'action_id': action_id, 'reward': reward,
                                'expert_weights': expert_weights, 'expert_mean_rewards': expert_mean_rewards}
        if self.record_best_expert:
            max_val = np.max(expert_mean_rewards)
            best_expert_ids = []
            for expert_id, expert_mean_reward in enumerate(expert_mean_rewards):
                if abs(expert_mean_reward - max_val) < 1e-6:
                    best_expert_ids.append(expert_id)
            self.history[self.t].update({'best_expert_ids': best_expert_ids,
                                         'best_expert_mean_reward': max_val})
        self.t += 1

    def save_history_to_csv(self, file_dir='', filename='history', timestamp=False):
        if timestamp:
            filename += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if '.csv' not in filename:
            filename += '.csv'
        with open(file_dir + filename, mode='w') as file:
            fieldnames = ['time', 'action_id', 'reward', 'expert_weights', 'expert_mean_rewards']
            if self.record_best_expert:
                fieldnames.extend(['best_expert_ids', 'best_expert_mean_reward'])
            writer = csv.writer(file, delimiter=',')
            writer.writerow(fieldnames)
            for t in range(1, len(self.history) + 1):
                record = self.history[t]
                if self.record_best_expert:
                    writer.writerow([t, record['action_id'], record['reward'],
                                     record['expert_weights'], record['expert_mean_rewards'],
                                     record['best_expert_ids'], record['best_expert_mean_reward']])
                else:
                    writer.writerow([t, record['action_id'], record['reward'],
                                     record['expert_weights'], record['expert_mean_rewards']])

    def save_actions_to_csv(self, file_dir='', filename='actions', timestamp=False):
        if timestamp:
            filename += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if '.csv' not in filename:
            filename += '.csv'
        with open(file_dir + filename, mode='w') as file:
            fieldnames = ['action_id', 'event_signal', 'name', 'comment', 'events_table']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for action_id, action in enumerate(self.actions):
                action_dict = {'action_id': action_id}
                campaign_start_date = list(action[0].keys())[0]
                action_description = action[0][campaign_start_date]['TimedEvents']
                if len(action_description) > 0:
                    for event_id in range(len(action_description)):
                        events_table = action_description[event_id]['events_table']
                        for entry_id in range(len(events_table)):
                            entry = events_table[entry_id]
                            date = list(entry.keys())[0]
                            events_table[entry_id] = {f'{date.month}/{date.day}': entry[date]}
                        action_dict.update(action_description[event_id])
                        writer.writerow(action_dict)
                else:
                    writer.writerow(action_dict)
