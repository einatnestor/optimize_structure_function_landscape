import numpy as np
from gym import Env, spaces
from scipy.integrate import solve_ivp
import copy
import landscape as ocg

# Constants:
SPECIES_NUM = 6
ACTION_NUM_PER_SPECIES = 2
OBSERVATION_SPACE = 2 ** SPECIES_NUM
NUM_OF_ITERATIONS = 200

def encode(strain, action):
    return strain * ACTION_NUM_PER_SPECIES + action


def decode(combined):
    strain = combined // ACTION_NUM_PER_SPECIES
    action = combined % ACTION_NUM_PER_SPECIES
    return strain, action

class landscape(ocg.landscape):

    def __init__(self,species_number, action_per_species, num_of_iterations, r_arr, alpha_arr, basel_production, interaction_production):
        super(landscape, self).__init__(species_number, action_per_species, num_of_iterations)
        self.r_arr = r_arr
        self.alpha_arr = alpha_arr
        self.dic_landscape = {}
        #self.A = A
        self.basel_production = basel_production
        self.interaction_production = interaction_production

    def reset_with_observation(self, observation):
        self.current_state = observation
        self.steps = 0
        return observation

    def convert_decimal_state_to_binary_arr(self, state):
        byte_format = '{0:0' + str(self.num_species) + 'b}'
        state = byte_format.format(state)
        state = np.array(list(state), dtype=int)
        return state


