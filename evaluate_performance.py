import copy
import os
import pickle
import numpy as np
import train_q_learning_discrete as ocg
import landscape_fixed_parameters as lfp
import operator
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import torch
import sys


SAVING_PERIOD = 100
ITERATION_LENGTH = 15
ACTION_PER_SPECIES = 2



def read_results(directory, SPECIES_NUMBER):
    # q_table = np.array(pd.read_csv(directory + "/q_table.csv").drop("Unnamed: 0", axis=1))
    reward_arr = np.array(pd.read_csv(directory + "/reward_per_episode.csv").drop("Unnamed: 0", axis=1)['0'])
    params = pickle.load(open(directory + '/params.pkl', 'rb'))

    r_values = np.reshape(np.array(pd.read_csv(directory + "/r_values.csv").drop("Unnamed: 0", axis=1)), SPECIES_NUMBER)
    alpha_values = np.array(pd.read_csv(directory + "/alpha_values.csv").drop("Unnamed: 0", axis=1))
    # A = np.reshape(np.array(pd.read_csv(directory + "/A.csv").drop("Unnamed: 0", axis=1)).transpose(), SPECIES_NUMBER)

    basel_production = np.reshape(
        np.array(pd.read_csv(directory + "/basel_production.csv").drop("Unnamed: 0", axis=1)).transpose(),
        SPECIES_NUMBER)
    interaction_production = np.reshape(
        np.array(pd.read_csv(directory + "/interaction_production.csv").drop("Unnamed: 0", axis=1)).transpose(),
        (SPECIES_NUMBER, SPECIES_NUMBER))
    env = lfp.landscape(SPECIES_NUMBER, ACTION_PER_SPECIES, 400, r_values, alpha_values, basel_production,
                        interaction_production)
    return reward_arr, r_values, env, params


def plot_reward(reward_arr, directory):
    # plt.plot([i for i in range(len(reward_arr))], reward_arr)
    # plt.title("Reward")
    # plt.ylabel("Reward per episode")
    # plt.xlabel("Episode")
    # plt.show()

    sliding_avg_reward = pd.DataFrame(reward_arr).rolling(100).mean()
    plt.plot([i for i in range(sliding_avg_reward.shape[0])], sliding_avg_reward)
    plt.title("Reward")
    plt.ylabel("Moving average of reward")
    plt.xlabel("Episode")
    if not os.path.isdir(directory + "/figures"):
        os.mkdir(directory + "/figures")
    plt.savefig(directory + "/figures" + "/Moving_average_of_reward.png")
    plt.show()


def compare_q_tables(directory, env, params, model_type, iterations, n):
    sns.set_theme(style="ticks", palette="pastel")
    starting_states = [i for i in range(n)]  #random.sample(range(2 ** SPECIES_NUMBER), n)  #
    # results_delta = []
    results_end = []
    iteration_arr = []
    delta_df = pd.DataFrame({"value": [], "episode": []})
    value_df = pd.DataFrame({"value": [], "episode": []})
    flag = True
    file_ext = ".csv" if model_type == "q_learning" else ".pkl"
    directory_name = "q_tables" if model_type == "q_learning" else "prev_agents"
    for iteration in range(SAVING_PERIOD, iterations + 50, SAVING_PERIOD):
        models_directory = directory + "/" + directory_name + "/" + str(iteration) + file_ext
        iteration_arr.append(iteration)
        if model_type == 'q_learning':
            model = np.array(pd.read_csv(models_directory).drop(['Unnamed: 0'], axis=1))
        else:
            model = pickle.load(open(models_directory, 'rb'))
        delta_iterations_values = []
        end_iterations_values = []
        start_iterations_values = []

        for state in starting_states:
            env.reset_with_observation(env.convert_decimal_state_to_binary_arr(state))
            value_start = env.reward()
            if flag:
                df_start_delta = pd.DataFrame({"value": 0, "episode": 0}, index=[0])
                delta_df = pd.concat([delta_df, df_start_delta], ignore_index=True)

                df_start = pd.DataFrame({"value": value_start, "episode": 0}, index=[0])
                value_df = pd.concat([value_df, df_start], ignore_index=True)
                start_iterations_values.append(value_start)

            value = simulate_episode(iteration,state,env, model_type, model)
            delta_iterations_values.append(value - value_start)
            end_iterations_values.append(value)
        if flag:
            results_end.append(start_iterations_values)
            flag = False

        # results_delta.append(delta_iterations_values)
        df_delta = pd.DataFrame()
        df_delta['value'] = pd.DataFrame(np.array(delta_iterations_values).transpose())
        df_delta['episode'] = np.array([iteration for i in range(n)]).transpose()
        delta_df = pd.concat([delta_df, df_delta], ignore_index=True)

        results_end.append(end_iterations_values)
        df_value = pd.DataFrame()
        df_value['value'] = pd.DataFrame(np.array(end_iterations_values).transpose())
        df_value['episode'] = np.array([iteration for i in range(n)]).transpose()
        value_df = pd.concat([value_df, df_value], ignore_index=True)


    results_end = pd.DataFrame(np.array(results_end).transpose())
    results_arr_sorted = results_end.sort_values(by=0, ascending=True)
    sns.lineplot(data=delta_df, x="episode", y="value")
    sns.despine()
    plt.savefig(directory + "/figures/change_from_starting_state.png", dpi=300)
    plt.show()
    #sns.heatmap(results_arr_sorted, cmap = "Blues")
    sns.heatmap(results_arr_sorted, cmap="Blues")
    plt.savefig(directory + "/figures/heatmap_last_clustered.png", dpi=300)
    plt.show()
    # sliding_avg_reward = pd.DataFrame(results_arr).diff(1,axis = 0)#pd.DataFrame(results_arr).rolling(2,axis= 1).apply(function)
    # sliding_avg_reward = np.array(sliding_avg_reward).mean(axis=0)
    # sliding_avg_reward['episode'] = [i for i in range(SAVING_PERIOD, 300, SAVING_PERIOD)]
    # sliding_avg_reward = sliding_avg_reward.melt()


def simulate_episode(iteration,state,env, type, model):

    for i in range(ITERATION_LENGTH - 1):
        curr_state = copy.deepcopy(env.get_state())
        curr_state_decimal = copy.deepcopy(env.decode_state(curr_state))
        curr_state = np.array([curr_state])
        if type == "q_learning":
            action = np.argmax(model[curr_state_decimal, :])
        elif type == "dqn":
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(curr_state, dtype=torch.float)))
                action = action.item()
        env.move(action)

    value = env.reward()

    return value


def main():

    SPECIES_NUMBER, ITERATIONS = int(sys.argv[1]), int(sys.argv[2])

    directory = "results/" + str(
        SPECIES_NUMBER) + "_dqn_species_2_actions_" + str(ITERATIONS) + "_episodes_" + str("10_03_2023_18_23")  # "_species_2_actions_10000_episodes_03_28_2023_18_10" #"_species_2_actions_10000_episodes_03_16_2023_12_06" #

    reward_arr, r_values, env, params = read_results(directory,SPECIES_NUMBER)
    # ocg.train(env, params, SPECIES_NUMBER, True)
    plot_reward(reward_arr, directory)
    compare_q_tables(directory, env, params, "dqn", ITERATIONS, 2 **SPECIES_NUMBER)


if __name__ == "__main__":
    main()
