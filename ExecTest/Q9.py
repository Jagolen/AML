
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import scipy.stats as stats

def s_to_t(s1, s2, std_t): #Function to get t|s1,s2,y with s1, s2 as given values
    return truncnorm.rvs(0 - (s1-s2)/std_t, (np.inf - (s1-s2)/std_t), loc = (s1-s2), scale = std_t)

def t_to_s(t, mu_s_vector, std_s_matrix, A, std_t): #Function to get s1,s2|t,y with t as a given value
    std_s_matrix_old = std_s_matrix
    std_s_matrix = np.linalg.inv(np.linalg.inv(std_s_matrix_old) + (A.reshape(-1, 1) * (1/std_t)) @ A)
    mu_s_vector = std_s_matrix@(np.linalg.inv(std_s_matrix_old) @ mu_s_vector + A.reshape(-1, 1) * (1/std_t)*t)
    mu_s_transpose = mu_s_vector.reshape(1, -1)[0]
    return np.random.multivariate_normal(mu_s_transpose, std_s_matrix)

def main():
    # initial
    mean_1 = 1
    var_1 = 4
    mean_2 = 1
    var_2 = 4
    beta = 5
    y = 1

    mu_s_vector = np.array([[mean_1], [mean_2]])
    std_s_matrix = np.array([[var_1, 0], [0, var_2]])
    A = np.array([[1, -1]]) # A-vector in the calculations

    Iters = 2000 #iterations
    ab = 30 # burnout period

    All_S = np.zeros((Iters, 2)) #Containing drawn S values
    All_t = np.zeros(Iters) #Containing drawn t values

    # Matrices where the s1 and s2 random values are stored as well as the µ-values (in All_Mu_S)
    All_S[0] = [np.random.normal(loc=mean_1, scale = var_1), np.random.normal(loc=mean_2, scale = var_2)]

    # We calculate t, then s, and repeat for as many times as the iterations
    for i in range(Iters-1):
        All_t[i] = s_to_t(All_S[i,0], All_S[i,1], beta)
        All_S[i+1] = t_to_s(All_t[i], mu_s_vector, std_s_matrix, A, beta)

    # Result vectors, x stores iterations, y stores s1 and s2, w stores µ1 and µ2
    x = [i for i in range(ab,Iters)]
    y1 = [All_S[i][0] for i in range(ab,Iters)]
    y2 = [All_S[i][1] for i in range(ab,Iters)]
    #Finding mean
    mu_S1 = np.mean(y1)
    mu_S2 = np.mean(y2)

    #Finding the Covariance matrix
    co_11 = 0
    co_121 = 0
    co_22 = 0

    for i in range(len(y1)):
        co_11 += (y1[i]-mu_S1)**2
        co_22 += (y2[i]-mu_S2)**2
        co_121 += (y1[i]-mu_S1)*(y2[i]-mu_S2)

    co_11 /= len(y1)-1
    co_121 /= len(y1)-1
    co_22 /= len(y2)-1
    samples_drawn = Iters - ab

    # Putting the results into a vector (µ) and a matrix (covariance)
    mu_vector = np.array([mu_S1, mu_S2])
    covariance_matrix = np.array([[co_11, co_121], [co_121, co_22]])
    # Draws samples from the original µ values and covariance matrix, the draws values from the new M values and covariance matrix
    All_samples_drawn = np.zeros((samples_drawn, 2))
    for i in range(samples_drawn):
        All_samples_drawn[i] = np.random.multivariate_normal(mu_vector, covariance_matrix)

    #Results are split into new s1, s2 and starting value for s1, s2
    newy1 = [All_samples_drawn[i][0] for i in range(samples_drawn)]
    newy2 = [All_samples_drawn[i][1] for i in range(samples_drawn)]

    # fc to t
    mean_fct = mean_1 - mean_2 #(np.array([1, -1])@np.array([[mean_1], [mean_2]]))[0]
    var_fct = beta + var_1 + var_2 #beta + np.array([1, -1])@np.array([[var_1, 0], [0, var_2]])@np.array([1, -1]).T

    # fd to t
    if y == 1:
        a, b = 0, 1000
    else:
        a, b = -1000, 0

    # q(t) approximation of p(t|y)
    a = (a - mean_fct)/np.sqrt(var_fct)
    b = (b - mean_fct)/np.sqrt(var_fct)
    mean_q = truncnorm.mean(a, b, loc=mean_fct, scale=np.sqrt(var_fct))
    var_q = truncnorm.var(a, b, loc=mean_fct, scale=np.sqrt(var_fct))

    # message passing algorithm

    # t to fc
    mean_tfc = (mean_q*var_fct - mean_fct*var_q)/(var_fct - var_q)
    var_tfc = (var_q*var_fct)/(var_fct - var_q)

    # fc to s1
    mean_fcs1 = mean_tfc + mean_2 #(np.array([1, 1])@np.array([[mean_tfc], [mean_2]]))[0]
    var_fcs1 = beta + var_tfc + var_2 #beta + np.array([1, 1])@np.array([[var_tfc, 0],[0, var_2]])@np.array([1, 1]).T

    # p(s1|y)
    mean_s1Iy = (mean_fcs1*var_1 + mean_1*var_fcs1)/(var_fcs1 + var_1)
    var_s1Iy = (var_fcs1*var_1)/(var_1 + var_fcs1) 

    # fc to s2
    mean_fcs2 = (np.array([1, -1])@np.array([[mean_tfc], [var_tfc]]))[0]
    var_fcs2 = beta + np.array([1, -1])@np.array([[var_tfc, 0], [0, var_1]])@np.array([1, -1]).T

    # p(s2|y)
    mean_s2Iy = (mean_fcs2*var_2 + mean_2*var_fcs2)/(var_fcs2 + var_2)
    var_s2Iy = var_fcs2*var_2/(var_2 + var_fcs2)

    # plot

    # posteriors
    x = np.linspace(-10, 10, num=1000)
    plt.plot(x, stats.norm.pdf(x, mu_S1, co_11))
    plt.plot(x, stats.norm.pdf(x, mu_S2, co_22))
    plt.hist(newy1, density=1, bins=20)
    plt.hist(newy2, density=1, bins=20)
    plt.plot(x, stats.norm.pdf(x, mean_s1Iy, var_s1Iy))
    plt.plot(x, stats.norm.pdf(x, mean_s2Iy, var_s2Iy))
    plt.legend(["S1 (Gibbs)", "S2 (Gibbs)", "S1 (Message)", "S2 (Message)","S1 Gibbs Histogram", "S2 Gibbs Histogram"])
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.title("Comparison of skill distributions")
    plt.show()
    

   
    print(f'mean_fct = {mean_fct}')
    print(f'var_fct = {var_fct}')
    print(f'mean_q = {mean_q}')
    print(f'var_q = {var_q}')
    print(f'mean_tfc = {mean_tfc}')
    print(f'var_tfc = {var_tfc}')
    print(f'mean_fcs1 = {mean_fcs1}')
    print(f'var_fcs1 = {var_fcs1}')
    print(f'mean_s1Iy = {mean_s1Iy}')
    print(f'var_s1Iy = {var_s1Iy}')
    print(f'mean_fcs2 = {mean_fcs2}')
    print(f'var_fcs2 = {var_fcs2}')
    print(f'mean_s2Iy = {mean_s2Iy}')
    print(f'var_s2Iy = {var_s2Iy}')
    




if __name__ == '__main__':
    main()