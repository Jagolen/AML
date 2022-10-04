from msilib.schema import Class
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, truncnorm

input_data = pd.read_csv("AML\Q6\SerieA.csv")

class team:
    def __init__(self, mu, var, name):
        self.mu = mu
        self.var = var
        self.name = name

def s_to_t(s1, s2, std_t): #Function to get t|s1,s2,y with s1, s2 as given values
    return truncnorm.rvs(0 - (s1-s2)/std_t, (np.inf - (s1-s2)/std_t), loc = (s1-s2), scale = std_t)

def t_to_s(t, A): #Function to get s1,s2|t,y with t as a given value
    global std_s_matrix
    global mu_s_vector
    std_s_matrix_old = std_s_matrix
    std_s_matrix = np.linalg.inv(np.linalg.inv(std_s_matrix_old) + (A.reshape(-1, 1) * (1/std_t)) @ A)
    mu_s_vector = std_s_matrix@(np.linalg.inv(std_s_matrix_old) @ mu_s_vector + A.reshape(-1, 1) * (1/std_t)*t)
    mu_s_transpose = mu_s_vector.reshape(1, -1)[0]
    return np.random.multivariate_normal(mu_s_transpose, std_s_matrix), [mu_s_vector[0][0], mu_s_vector[1][0]]


team_list = []
for teams in input_data["team1"]:
    if teams not in team_list:
        team_list.append(teams)

for teams in input_data["team2"]:
    if teams not in team_list:
        team_list.append(teams)

print(team_list)

team_data = []
for i in team_list:
    team_data.append(team(1,4,i))