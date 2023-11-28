# optimize_structure_function_landscape

Using reinforcment learning to optimize the function of a structre-function landscape.
Input:  N interacting species
        The dynamics of the species (will be used to initiate the environment)
Output: The optimization path (the result in the maximum function value) for every initial community composition.

Environment:
State - binary vector in length N, represent the presence/ absance of each species in the commnunity.
Action - Either add or remove one species at a step.
Reward - The function value. 


