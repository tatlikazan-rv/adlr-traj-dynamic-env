import torch
import numpy as np
from itertools import count
import pickle

from environment import GridWorldEnv
from training import init_model, select_action, obstacle_sort, select_action_smooth, Scheduler

from model import *
import json

from util_sacx.q_table import QTable
from util_sacx.policy import BoltzmannPolicy

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #m = input("Select normal Model (m) OR \n \
    #Model with pretrain (mp) OR \n \
    #Tests with number (#): ")
    m = "0"
    if m == "m":
        model_path = "SAC-X/model/"
    elif m == "mp":
        model_path = "SAC-X/model_pretrain/"
    elif m == "0":
        model_path = "SAC-X/test_models/Sa-t0/"
    elif m == "1":
        model_path = "SAC-X/test_models/Sa-t1/"
    elif m == "10":
        model_path = "SAC-X/test_models/Sa-t10/"
    elif m == "11":
        model_path = "SAC-X/test_models/Sa-t11/"
    elif m == "12":
        model_path = "SAC-X/test_models/Sa-t12/"
    elif m == "3":
        model_path = "SAC-X/test_models/Sa-t3/"
    elif m == "4":
        model_path = "SAC-X/test_models/Sa-t4/"
    elif m == "5":
        model_path = "SAC-X/test_models/Sa-t5/"
    elif m == "6":
        model_path = "SAC-X/test_models/Sa-t6/"
    elif m == "7":
        model_path = "SAC-X/test_models/Sa-t7/"
    elif m == "9":
        model_path = "SAC-X/test_models/Sa-t9/"
    elif m == "3000":
        model_path = "SAC-X/test_models/3000normaltrain_allfeatures/"
    elif m == "thc":
        model_path = "SAC-X/test_models/T-thc/"
    elif m == "thc1":
        model_path = "SAC-X/test_models/T-thc1/"
    elif m == "t0":
        model_path = "SAC-X/test_models/T-t0/"
    elif m == "t10":
        model_path = "SAC-X/test_models/T-t10/"
    elif m == "t11":
        model_path = "SAC-X/test_models/T-t11/"
    elif m == "t12":
        model_path = "SAC-X/test_models/T-t12/"
    elif m == "t3":
        model_path = "SAC-X/test_models/T-t3/"
    elif m == "t4":
        model_path = "SAC-X/test_models/T-t4/"
    elif m == "t5":
        model_path = "SAC-X/test_models/T-t5/"
    elif m == "t6":
        model_path = "SAC-X/test_models/T-t6/"
    elif m == "t7":
        model_path = "SAC-X/test_models/T-t7/"
    elif m == "t9":
        model_path = "SAC-X/test_models/T-t9/"


    # Load the model parameters
    with open(model_path + 'env_parameters.txt', 'r') as file:
        env_parameters = json.load(file)
    with open(model_path + 'hyper_parameters.txt', 'r') as file:
        hyper_parameters = json.load(file)
    with open(model_path + 'reward_parameters.txt', 'r') as file:
        reward_parameters = json.load(file)
    with open(model_path + 'feature_parameters.txt', 'r') as file:
        feature_parameters = json.load(file)

    tasks = (0, 1, 2, 3, 4)
    sac_schedule = Scheduler(tasks)

    with open(model_path + "Q_task.pkl", "rb") as tf:
        sac_schedule.Q_task.store = pickle.load(tf)

    # initialize environment
    env = GridWorldEnv(render_mode=None,
                       object_size=env_parameters['object_size'],  # TODO: change back to env_size to radius objects
                       num_obstacles=env_parameters['num_obstacles'],
                       window_size=env_parameters['window_size'],
                       reward_parameters=reward_parameters,
                       env_parameters=env_parameters)
    env.render_mode = "human"

    # initialize NN
    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model(hyper_parameters["input_dims"])
    seed = feature_parameters['seed_init_value'] + 10
    # Load model
    actorNet.load_state_dict(torch.load(model_path + "actor.pt", map_location=device))
    # actorNet.max_sigma = hyper_parameters['sigma_final']
    criticNet_1.load_state_dict(torch.load(model_path + "criticNet_1.pt", map_location=device))
    criticNet_2.load_state_dict(torch.load(model_path + "criticNet_2.pt", map_location=device))
    target_valueNet.load_state_dict(torch.load(model_path + "target_valueNet.pt", map_location=device))

    if feature_parameters['apply_environment_seed']:
        seed = 0  # feature_parameters['seed_init_value']fixed
    action_history = deque(maxlen=feature_parameters['action_history_size'])

    seed = feature_parameters['seed_init_value'] + 70
    # + 25 - 2 success
    task = 0
    i_episode = 0
    while i_episode < 10:  # run plot for 10 episodes to see what it learned
        List_Tau = []
        if i_episode == 0 or i_episode == 1:
            entropy_factor = hyper_parameters['entropy_factor']
            sigma_ = hyper_parameters['sigma_init']
        else:
            entropy_factor = hyper_parameters['entropy_factor'] + i_episode * (
                    hyper_parameters['entropy_factor_final'] - hyper_parameters['entropy_factor']) / (
                                     hyper_parameters["num_episodes"] - 1)
            sigma_ = hyper_parameters['sigma_init'] + i_episode * (
                    hyper_parameters['sigma_final'] - hyper_parameters['sigma_init']) / (
                             hyper_parameters["num_episodes"] - 1)
        actorNet.max_sigma = sigma_
        i_episode += 1
        # Initialize the environment and state
        if feature_parameters['apply_environment_seed']:
            env.reset(seed=seed)
            seed += 1
        else:
            env.reset()
        obs = env._get_obs()
        if feature_parameters['sort_obstacles']:
            obs = obstacle_sort(obs)
        obs_values = np.append(obs["agent"], obs["target"])
        for idx_obstacle in range(env_parameters['num_obstacles']):
            obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
        state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

        state = state.view(1, -1)
        for t in count():  # every step of the environment
            # Select and perform an action
            if t % sac_schedule.xi == 0:
                fuzzy_state = sac_schedule.caluculate_fuzzy_distance(obs_values)
                # task = random.choice(tasks) ## sac-u
                task = sac_schedule.schedule_task(List_Tau, fuzzy_state)  ## sac-q
                print(task)
                List_Tau.append(task)

            action = select_action(state, actorNet, task)
            if feature_parameters['action_smoothing']:
                action_history.extend([action])
                action = select_action_smooth(action_history)
            _, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward[task]], dtype=torch.float, device=device)

            # Observe new state
            obs = env._get_obs()
            if not done:
                if feature_parameters['sort_obstacles']:
                    obs = obstacle_sort(obs)
                obs_values = np.append(obs["agent"], obs["target"])
                for idx_obstacle in range(env_parameters['num_obstacles']):
                    obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
                next_state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

                next_state = next_state.view(1, -1)
            else:
                next_state = None
                break

            # Store the transition in memory
            action = np.array([action])
            action = torch.tensor(action, dtype=torch.float).to(actorNet.device)
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
