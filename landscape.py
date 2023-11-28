import numpy as np
from gym import Env, spaces
from scipy.integrate import solve_ivp
import copy
import torch


###########################################################################################################
# This is the most simple version: Discrete states & actions, Basic LV dynamics and linear value function.#
# In general, the states are binary vectors (x_i is the presence if species i)                            #
# Action are removing species ot adding species                                                           #
###########################################################################################################


class landscape(Env):
    def __init__(self, species, action_per_species, num_of_iterations):
        super(landscape, self).__init__()
        self.tf = 60
        self.num_species = species
        self.num_of_iterations = num_of_iterations
        self.action_per_species = action_per_species
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_species,), dtype=np.int)
        self.action_space = spaces.Discrete(self.num_species * action_per_species)
        self.action_dic = {0: "remove", 1: "add"}
        self.steps = 0
        self.r_arr = np.random.uniform(0.1, 1, size=self.num_species)
        self.alpha_arr = np.random.uniform(0, 1.5, size=(self.num_species, self.num_species))
        self.basel_production = np.random.uniform(0, 2, size=self.num_species)  # todo check range
        self.interaction_production = np.random.uniform(0, 5,
                                                        size=(self.num_species, self.num_species))  # todo check range
        # self.A = np.random.uniform(1, 50 , size=self.num_species)
        self.current_state = None

    def reset(self, ):
        observation = self.observation_space.sample()#np.zeros(self.num_species)
        self.current_state = observation
        self.steps = 0
        return observation

    def get_number_of_species(self):
        return self.num_species

    def function(self, x):
        # return np.sum(np.dot(self.A, x))
        return np.dot(self.basel_production, x) + np.dot(x, np.dot(self.interaction_production, x))

    def LV(self, t, x):
        r_for_multiplication = np.full((self.num_species, self.num_species), self.r_arr).transpose()
        # x_dot = self.r_arr * x * (1 - np.dot(self.r_arr * self.alpha_arr, x))
        x_dot = self.r_arr * x * (1 - np.dot(self.alpha_arr, x))
        # thr = 1e-10  # Set a threshold
        # x_dot[x < thr] = 0
        return x_dot

    def index_event(self, i, th=10 ** (-6)):
        def set_to_zero(t, y):
            return y[i] - th

        # set_to_zero.terminal = False
        return set_to_zero

    def reward(self):
        x = np.full(self.num_species, 0.1) * self.current_state
        abundance = self.simulate(x)
        value = self.function(abundance)
        return value

    def get_state(self):
        return copy.deepcopy(self.current_state)

    def simulate(self, x, ):
        func = lambda t, y0: self.LV(t, y0)
        evs = [self.index_event(i) for i in range(len(x))]
        solution = solve_ivp(func, (0, self.tf), y0=x, events=evs)
        y = np.transpose(solution.y)[-1]
        return y

    def simulate_dynamics(self, state):
        state_t = torch.tensor(state)
        state_t = state_t.numpy()
        x = np.full(self.num_species, 0.1) * state_t
        if len(x.shape) >1:
            x = x[0]
        abundance = self.simulate(x)
        return abundance

    def decode_state(self, state):
        str_state = ""
        for letter in state:
            str_state += str(letter)
        return int(str_state, 2)

    def move(self, action):
        strain, step = self.decode(action)
        if self.action_dic[step] == "remove":
            self.current_state[strain] = 0
        else:  # addition
            self.current_state[strain] = 1

    def step(self, action):
        self.steps += 1
        done = False
        self.move(action)
        reward_new = self.reward()
        reward = reward_new
        if self.steps >= self.num_of_iterations:
            done = True
        return copy.deepcopy(self.current_state), reward, done, []

    def encode(self, strain, action):
        return strain * self.action_per_species + action

    def decode(self, combined):
        strain = combined // self.action_per_species
        action = combined % self.action_per_species
        return strain, action
