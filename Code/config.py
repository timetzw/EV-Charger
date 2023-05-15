import numpy as np
# Variables for Choice Model
alpha = -0.5
beta = 1.2
gamma = -2

# Variables for Simulation Model (Staying Probability)
leave_prob = 0.3

# Variables for Report Model
theta = 4

# Global Variables
num_cars = 70  # the number of cars (drivers)
# inflow_ratio =
observation_time = 40  # total time observed

total_parking_lot = 2  # the number of parking lots
total_chargers = 20  # total number of charger
upper_bound_per_parking_lot = 20  # max number in each parking lots (limits)
charging_time_base = 30  # the base of charging time
charging_time_range = 10



# Experiments
dist_base = 60
dist_change = 60
price_base = 5
price_change = 5
charging_time_switch = 0  # 0 as fixed, 1 as random

# Game Day Settings
price_gameday_multiple = 10
dist_gameday_multiple = 0.3


# Randomness control
seed = 19999  # should not be 0
# seed = np.random.randint(2**32)

# Scenario Variable
scenario = 0  # 0 as normal, otherwise it is the "k"-th parking lot having GAME (in code, it is PL k-1, in reality,
# it is k)

# Random Control
price_first_data_randomness_control = 0  # 0 as not random, while others as seed
dist_first_data_randomness_control = 0
arrive_first_data_randomness_control = 0

leave_control = 0  # 0 as not activated, others as activated

# Figure
x_stick_on = True