import numpy as np
import scipy.stats as sc
import pandas as pd
import random

from sympy import total_degree

def sigma_matrix(sig_mat, sigma_t, AtA): # function to calculate Sigma_a|b
    sigma_A = np.linalg.inv(sig_mat) + 1/sigma_t*AtA
    sigma_A = np.linalg.inv(sigma_A)
    return sigma_A

def trunc(s1, s2, sigma_t, y): # return simulated value of t
    # My mean and standard deviation for the truncated normal
    my_mean = s1-s2
    my_std = np.sqrt(sigma_t)

    # Find boundaries of standardized truncated normal
    inf = 1000
    if y > 0:
        myclip_a = 0
        a = (myclip_a - my_mean) / my_std
        b = inf
    else:
        myclip_b = 0
        b = (myclip_b - my_mean) / my_std
        a = -inf

    X = sc.truncnorm(a,b) # Standardized truncated normal
    value = X.rvs() # Draw a value from the standardized truncated normal distribution
    value = my_mean + value*my_std # Convert to my truncated normal distribution
    return value

def Mu_arr(Sigma_ab, sig_mat, mu_mat, At, sigma_t,t): # function to calculate mu_a|b
    mu_ab = np.dot(Sigma_ab,(np.dot(np.linalg.inv(sig_mat),mu_mat) + 1/sigma_t*At*t))
    return mu_ab

    
def s_val(Sigma_ab,i): # function to calculate value of S, i = 1 or 2
    if i == 1:
        S = Sigma_ab[0,0] - Sigma_ab[0,1]*Sigma_ab[1,1]**(-1)*Sigma_ab[1,0]
    elif i == 2:
        S = Sigma_ab[1,1] - Sigma_ab[1,0]*Sigma_ab[0,0]**(-1)*Sigma_ab[0,1]
    else:
        print("ERROR")
        return 0
    return S

    
def draw_val(sigma_A, mu_ab, s, i): # get value of s1 or s2
    m = m_val(sigma_A, mu_ab, s, i)
    s = s_val(sigma_A, i)
    return np.random.normal(m,np.sqrt(s))

def m_val(Sigma_ab, mu_ab, s, i): # function to calculate of m
    if i == 1:
        m = mu_ab[0] + Sigma_ab[0,1]*Sigma_ab[1,1]**(-1)*(s-mu_ab[1])
    elif i == 2:
        m = mu_ab[1] + Sigma_ab[1,0]*Sigma_ab[0,0]**(-1)*(s-mu_ab[0])

    return m

def gibbs(s1, s2, t, sigma_A, At, sigma_t, y, sig_mat, mu_mat, N):
    t[0] = trunc(s1[0],s2[0],sigma_t,y)
    for i in range(N):
        mu_ab = Mu_arr(sigma_A, sig_mat, mu_mat, At, sigma_t, t[i])
        s1[i+1] = draw_val(sigma_A, mu_ab, s2[i], 1)
        s2[i+1] = draw_val(sigma_A, mu_ab, s1[i+1], 2)
        t[i+1] = trunc(s1[i+1], s2[i+1], sigma_t, y)

def prediction(mu1, mu2, std1, std2, y):
    #will try to predict N simulations of given match and return how many are correct
    N=100
    t=np.zeros(N)
    mut=np.random.normal(mu1,std1,N)-np.random.normal(mu2,std2,N)
    for i in range(N):
        t[i]=np.random.normal(mut[i],5)
    accuracy=sum(y==np.sign(t))/N
    return accuracy


def rank_predictor_guess(teams, outcome, team1, team2, teams_mean, teams_var, burn_in, n_samples):

    n_games = len(np.array(team1))

    # A matrix
    A = np.array([1, -1]) 
    At = np.array([[1],[-1]])
    AtA = A*At
    accuracy, total_games = 0, 0

    # Loop over all games
    for game in range(n_games):
        print(f"{game} of {n_games}: {(game/n_games*100):.2f}%")

        y = outcome[game]
        if y != 0:
            # Get mean and variance of the teams
            home_team = teams[team1[game]]
            away_team = teams[team2[game]]
            sigma_1 = teams_var[home_team]
            sigma_2 = teams_var[away_team]
            sigma_t = 2 #sigma_1 + sigma_2
            mu_1 = teams_mean[home_team]
            mu_2 = teams_mean[away_team]
            
            # Calculate the amount of correct guesses of result
            total_games = total_games + 1
            accuracy += prediction(mu_1, mu_2, sigma_1, sigma_2, y)
                
            

            # Get some necessary matrices
            sig_mat = np.array([[sigma_1, 0],[0, sigma_2]])
            mu_mat = np.array([[mu_1],[mu_2]])
            Sigma_ab = sigma_matrix(sig_mat, sigma_t, AtA)

            # Implement Gibbs sampler
            s1 = np.zeros(n_samples+1)
            s2 = np.zeros(n_samples+1)
            t = np.zeros(n_samples+1)
            gibbs(s1,s2,t,Sigma_ab,At,sigma_t,y,sig_mat,mu_mat,n_samples)

            # Update mean and variance of the teams
            teams_mean[home_team] = np.mean(s1[burn_in:])
            teams_mean[away_team] = np.mean(s2[burn_in:])
            teams_var[home_team] = np.var(s1[burn_in:])
            teams_var[away_team] = np.var(s2[burn_in:])
    
    return accuracy/total_games # return r



def main():
    column_names = ['date', 'time', 'team1', 'team2', 'score1', 'score2']
    data = pd.read_csv("ExecTest/SerieA.csv", names=column_names, skiprows=1)

    team1 = np.array(data.team1.tolist())
    team2 = np.array(data.team2.tolist())

    result = np.sign(np.array(data.score1.tolist()) - np.array(data.score2.tolist()))
    nr_games = len(np.array(team1))

    unique_teams = np.unique(np.concatenate((team1, team2)))

    teams = {}
    for idx in range(len(unique_teams)):
        teams[unique_teams[idx]] = idx

    nr_of_teams = len(unique_teams)

    team_mean = np.zeros(nr_of_teams)
    team_var = np.ones(nr_of_teams)

    burn_in = 2
    N_samples = 100

    # Original order of games
    prediction_rate_original = rank_predictor_guess(teams, result, team1, team2, team_mean, team_var, burn_in, N_samples)


    # Shuffled order of games
    original_order = list(range(nr_games))
    random.shuffle(original_order)
    result = [result[i] for i in original_order]
    team1 = [team1[i] for i in original_order]
    team2 = [team2[i] for i in original_order]
    prediction_rate_shuffled = rank_predictor_guess(teams, result, team1, team2, team_mean, team_var, burn_in, N_samples)

    # Random guessing
    correct_guess, total_games = 0, 0
    for i in range(nr_games):
        if result[i] != 0:
            total_games += 1
            if (random.randint(0,1)*2)-1 == result[i]:
                correct_guess += 1
    prediction_rate_random = correct_guess/total_games

    print(f"Prediction Rate with orginal order of games: r={prediction_rate_original:.5f}")
    print(f"Prediction Rate with shuffled order of games: r={prediction_rate_shuffled:.5f}")
    print(f"Prediction Rate with random guessing of games: r={prediction_rate_random:.5f}")


main()