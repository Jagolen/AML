import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

#Class for every team, showing µ, σ and team name
class team:
    def __init__(self, mu, var, name, skill):
        self.mu = mu
        self.var = var
        self.name = name
        self.skill = skill

def s_to_t(s1, s2, std_t, score): #Function to get t|s1,s2,y with s1, s2 as given values
    if score == -1:
        minval, maxval = -np.inf, 0
    else:
        minval, maxval = 0, np.inf
    return truncnorm.rvs(minval - (s1-s2)/std_t, (maxval - (s1-s2)/std_t), loc = (s1-s2), scale = std_t)

def t_to_s(t, A, std_t,std_s_matrix, mu_s_vector): #Function to get s1,s2|t,y with t as a given value
    std_s_matrix_old = std_s_matrix
    std_s_matrix = np.linalg.inv(np.linalg.inv(std_s_matrix_old) + (A.reshape(-1, 1) * (1/std_t)) @ A)
    mu_s_vector = std_s_matrix@(np.linalg.inv(std_s_matrix_old) @ mu_s_vector + A.reshape(-1, 1) * (1/std_t)*t)
    mu_s_transpose = mu_s_vector.reshape(1, -1)[0]
    return np.random.multivariate_normal(mu_s_transpose, std_s_matrix), mu_s_vector, std_s_matrix

std_t = 5 # starting σ for t

#Reads the data, the path has to be changed depending on where your target folder is in vs code, here the target folder is AML
input_data = pd.read_csv("Q10/res_fence_short.csv")

input_data

#Creates list of names
team_list = []
for teams in input_data["ID1"]:
    if teams not in team_list:
        team_list.append(teams)

for teams in input_data["ID2"]:
    if teams not in team_list:
        team_list.append(teams)

#Creates a list with all the teams (i.e. a list of classes)
team_data = []
for i in team_list:
    team_data.append(team(1,4,i,0))

Iters = 2000 #Iterations
ab = 500 #After burnout
A = np.array([[1, -1]]) # A-vector in the calculations

#Main loop
nr_matches = len(input_data["ID1"])
for iteration in range(nr_matches):
    print(f"Calculating match {iteration+1} of {nr_matches}")
    i = iteration # Doing matches in sequence
    #i = nr_matches-1-iteration # doing matches in reverse
    #Get the score, if score is 0, the match is ignored
    score = input_data["PD"][i]
    if score == 0:
        continue

    #Only the sign of the score is used
    score = int(score)
    score = np.sign(score)

    #Get the correct team data by fetching the names and then using the classes corresponding to those names
    team_1_name = input_data["ID1"][i]
    team_2_name = input_data["ID2"][i]

    for j in range(len(team_data)):
        if team_data[j].name == team_1_name:
            team1 = team_data[j]
        if team_data[j].name == team_2_name:
            team2 = team_data[j]
            
    #Set up µ-vector and covariance matrix
    mu_s_vector = np.array([[team1.mu], [team2.mu]])
    std_s_matrix = np.array([[team1.var**2, 0], [0, team2.var**2]])

    All_S = np.zeros((Iters, 2)) #Containing drawn S values
    All_t = np.zeros(Iters) #Containing drawn t values

    # Matrices where the s1 and s2 random values are stored as well as the µ-values (in All_Mu_S)
    All_S[0] = [np.random.normal(loc=team1.mu, scale = team1.var), np.random.normal(loc=team2.mu, scale = team2.var)]

    # We calculate t, then s, and repeat for as many times as the iterations
    for j in range(Iters-1):
        All_t[j] = s_to_t(All_S[j,0], All_S[j,1], std_t, score)
        All_S[j+1], mu_s_vector, std_s_matrix = t_to_s(All_t[j], A, std_t,std_s_matrix, mu_s_vector)

    #Getting s1 and s2 values
    y1 = [All_S[j][0] for j in range(ab,Iters)]
    y2 = [All_S[j][1] for j in range(ab,Iters)]

    #Getting µ and σ
    mu_team1 = np.mean(y1)
    mu_team2 = np.mean(y2)
    var_team1 = 0
    var_team2 = 0
    for j in range(len(y1)):
        var_team1 += (y1[j]-mu_team1)**2
        var_team2 += (y2[j]-mu_team2)**2
    var_team1 /= len(y1)-1
    var_team2 /= len(y1)-1
    var_team1 = math.sqrt(var_team1)
    var_team2 = math.sqrt(var_team2)
    team1.mu = mu_team1
    team2.mu = mu_team2
    team1.var = var_team1
    team2.var = var_team2

#Drawing 100 random numbers from the distributions to determine skill
for current_team in team_data:
    for i in range(100):
        partial_skill = np.random.normal(loc=current_team.mu, scale = current_team.var)/100
        current_team.skill += partial_skill

team_data.sort(key=lambda team: team.skill, reverse=True)

print("TEAM SKILLS AND RANKINGS:")
for i in range(len(team_data)):
    print(f"{i+1}. {team_data[i].name}: {round(team_data[i].skill, 2)}")

    





