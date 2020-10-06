import sys

sys.path.append('..')

from tqdm import tqdm

from src.models import *
from src.actions import AgroActions


# Inputs
RECORD_BEST_EXPERT = True
TIME_HORIZON = 1000
DELTA = .1
RHO = None
ACTIONS, COSTS = AgroActions().create_actions([0, 1, 4, 7], [0, 15], year=2019)  # Year will be overridden
EXPERTS = ([UniformExpert(ACTIONS)]
           + [Expert(ACTIONS) for _ in range(3)]
           + [NoActionExpert(ACTIONS)]
           + [WofostExpert(ACTIONS)]
           + [PerturbedWofostExpert(ACTIONS, .01)]
           + [PerturbedWofostExpert(ACTIONS, .1)]
           + [ComplementaryWofostExpert(ACTIONS)]
           + [MinWofostExpert(ACTIONS)]
           + [MaxWofostExpert(ACTIONS)]
           + [PerturbedMaxWofostExpert(ACTIONS, .01)]
           + [PerturbedMaxWofostExpert(ACTIONS, .1)])

# Initialization
env = Environment(ACTIONS, EXPERTS, RECORD_BEST_EXPERT)
exp4r = Exp4R(ACTIONS, EXPERTS, TIME_HORIZON, DELTA, RHO)

# Algorithm
for _ in tqdm(range(TIME_HORIZON)):
    env.update()
    exp4r.get_advice()
    exp4r.combine_advice()
    action_id = exp4r.sample_action_id()
    reward = env.reward_wofost(action_id)
    env.add_history(action_id, reward, exp4r.expert_weights, exp4r.advice)
    exp4r.update_weights(action_id, reward)

env.save_history_to_csv()
env.save_actions_to_csv()

print('Simulation done!')
exp4r.print_regret_upper_bound()
exp4r.threshold_test()
