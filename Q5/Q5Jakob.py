from statistics import covariance
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, truncnorm

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

time_taken = time.time()
mu_s = 1
std_s = 4
mu_s_start = np.array([mu_s, mu_s])
mu_s_vector = np.array([[mu_s], [mu_s]])
std_s_matrix = np.array([[std_s**2, 0], [0, std_s**2]])
std_s_start = np.array([[std_s**2, 0], [0, std_s**2]])
A = np.array([[1, -1]])
std_t = 5

Iters = 10000 #iterations
ab = 500 # burnout period
All_S = np.zeros((Iters, 2)) #Containing drawn S values
All_t = np.zeros(Iters) #Containing drawn t values
All_Mu_S = np.zeros((Iters, 2)) #mean of S in every iteration step

All_S[0] = [np.random.normal(loc=mu_s, scale = std_s), np.random.normal(loc=mu_s, scale = std_s)]
All_Mu_S[0] = [mu_s_vector[0][0], mu_s_vector[1][0]]

for i in range(Iters-1):
    All_t[i] = s_to_t(All_S[0,0], All_S[0,1], std_t)
    All_S[i+1], All_Mu_S[i+1] = t_to_s(All_t[i], A)

x = [i for i in range(ab,Iters)]
y1 = [All_S[i][0] for i in range(ab,Iters)]
y2 = [All_S[i][1] for i in range(ab,Iters)]
w1 = [All_Mu_S[i][0] for i in range(ab,Iters)]
w2 = [All_Mu_S[i][1] for i in range(ab,Iters)]


# How the mean changes (after burnout period)
plt.plot(x,w1)
plt.plot(x,w2)
plt.legend(["µ_1", "µ_2"])
plt.xlabel("Iterations")
plt.ylabel("µ-value")
plt.show()

#FINDING NEW MEAN AND CO VARIANCE FROM DRAWN S1 AND S2 FOR POSTERIOR DISTRIBUTION

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

mu_vector = np.array([mu_S1, mu_S2])
covariance_matrix = np.array([[co_11, co_121], [co_121, co_22]])

samples_drawn = Iters - ab

All_samples_drawn = np.zeros((samples_drawn, 2))
Samples_from_start = np.zeros((samples_drawn, 2))
for i in range(samples_drawn):
    All_samples_drawn[i] = np.random.multivariate_normal(mu_vector, covariance_matrix)
    Samples_from_start[i] = np.random.multivariate_normal(mu_s_start, std_s_start)

newy1 = [All_samples_drawn[i][0] for i in range(samples_drawn)]
newy2 = [All_samples_drawn[i][1] for i in range(samples_drawn)]

start1 = [Samples_from_start[i][0] for i in range(samples_drawn)]
start2 = [Samples_from_start[i][1] for i in range(samples_drawn)]

time_taken = time.time() - time_taken
print(f"Time taken = {time_taken}")

figure, axis = plt.subplots(1,2)
axis[0].hist(y1, bins=20)
axis[0].hist(newy1, bins=20)
axis[0].legend(["s_1 (Gibbs)", "s_1 (Distribution)"])

axis[1].hist(y2, bins=20)
axis[1].hist(newy2, bins=20)
axis[1].legend(["s_2 (Gibbs)", "s_2 (Distribution)"])

plt.xlabel("s-value")
plt.ylabel("Occurences")
plt.title(f"{Iters} iterations, took {round(time_taken, 2)} seconds")

plt.show()

figure, axis = plt.subplots(1,2)
axis[0].hist(start1, bins='auto')
axis[0].hist(newy1, bins='auto')
axis[0].legend(['s_1', 's_1|y'])

axis[1].hist(start2, bins='auto')
axis[1].hist(newy2, bins='auto')
axis[1].legend(['s_2', 's_2|y'])

plt.xlabel("s-value")
plt.ylabel("Occurences")
plt.show()





""" 
MATLAB:

my1=1;
my2=-1;
sigma1=1;
sigma2=4;
beta=5;

sum= inv(inv([sigma1 0;0 sigma2])+[1 -1]'*(1/beta)*[1 -1]);
my = sum*(inv([sigma1 0; 0 sigma2])*[my1; my2]+[1 -1]'*(1/beta).*(3)) """



""" 
ELIAS

mu_ss = [1, -1]
E_ss = np.array([[1, 0], [0, 4]])
E_tIss = 5
t = 3

def sonty(E_ss, E_tIss, mu_ss, t): # from q3.1
    E_ssIt = np.linalg.inv(np.linalg.inv(E_ss)+1/E_tIss) #np.linalg.inv(E_bIa))
    mu_ssIt = E_ssIt@(np.linalg.inv(E_ss)@mu_ss + np.array([1, -1]).T*np.array([1/E_tIss*1]))
    return [mu_ssIt, E_ssIt]

def tonsy(E_ss, E_tIss, mu_ss, t): # from q3.2
    A = np.array([1, -1])
    mu_t = A@mu_ss
    E_t = E_tIss+A@E_ss@A.T
    return [mu_t, E_t]


s_on_ty = [sonty(E_ss, E_tIss, mu_ss, t)]
t_on_sy = [tonsy(E_ss, E_tIss, mu_ss, t)]





print(s_on_ty, '\n')

for _ in range(2):
    pass#print(_)
    
    #print(s_on_ty[-1], '\n')
    
    
print(s_on_ty)
print(t_on_sy) """