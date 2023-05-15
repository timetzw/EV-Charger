import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
import sys
from funs import *
import config
# for scenario in range(4+1): # we only have 4 slots and I considered five (normal, game at 0,1,2,3) conditions
scenario = config.scenario
theta = config.theta

seed = config.seed
price_ctl = config.price_first_data_randomness_control
dist_ctl = config.dist_first_data_randomness_control
arrive_ctl = config.arrive_first_data_randomness_control

df = generate_first_data(slots, m, max_time, scenario, seed, price_ctl, dist_ctl, arrive_ctl)
file = open(f"./data/Exported Data_Scenario_{scenario}_theta_{theta}.txt".format(scenario, theta), "w")
r = report(df, file, scenario, seed)
r.main(m=mmax, slots=slots)
print("Seed was:", seed)
file.close()
