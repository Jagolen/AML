import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

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

#Timer
time_taken = time.time()

#Starting µ and σ for S
mu_s = 1
std_s = 4

#The start is a ROW since it is an input to a function that generates 2d normal distributions
# The vector is a COLUMN since it is used in calculations as was done in Q3
mu_s_start = np.array([mu_s, mu_s])
mu_s_vector = np.array([[mu_s], [mu_s]])
std_s_matrix = np.array([[std_s**2, 0], [0, std_s**2]])
std_s_start = np.array([[std_s**2, 0], [0, std_s**2]])
A = np.array([[1, -1]]) # A-vector in the calculations
std_t = 5 # σ for t

Iters = 2000 #iterations
ab = 500 # burnout period
All_S = np.zeros((Iters, 2)) #Containing drawn S values
All_t = np.zeros(Iters) #Containing drawn t values
All_Mu_S = np.zeros((Iters, 2)) #mean of S in every iteration step

# Matrices where the s1 and s2 random values are stored as well as the µ-values (in All_Mu_S)
All_S[0] = [np.random.normal(loc=mu_s, scale = std_s), np.random.normal(loc=mu_s, scale = std_s)]
All_Mu_S[0] = [mu_s_vector[0][0], mu_s_vector[1][0]]

# We calculate t, then s, and repeat for as many times as the iterations
for i in range(Iters-1):
    All_t[i] = s_to_t(All_S[i,0], All_S[i,1], std_t)
    All_S[i+1], All_Mu_S[i+1] = t_to_s(All_t[i], A)

# Result vectors, x stores iterations, y stores s1 and s2, w stores µ1 and µ2
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

# Putting the results into a vector (µ) and a matrix (covariance)
mu_vector = np.array([mu_S1, mu_S2])
covariance_matrix = np.array([[co_11, co_121], [co_121, co_22]])

# Samples excluding the burnout period
samples_drawn = Iters - ab

# Draws samples from the original µ values and covariance matrix, the draws values from the new M values and covariance matrix
All_samples_drawn = np.zeros((samples_drawn, 2))
Samples_from_start = np.zeros((samples_drawn, 2))
for i in range(samples_drawn):
    All_samples_drawn[i] = np.random.multivariate_normal(mu_vector, covariance_matrix)
    Samples_from_start[i] = np.random.multivariate_normal(mu_s_start, std_s_start)

#Results are split into new s1, s2 and starting value for s1, s2
newy1 = [All_samples_drawn[i][0] for i in range(samples_drawn)]
newy2 = [All_samples_drawn[i][1] for i in range(samples_drawn)]

start1 = [Samples_from_start[i][0] for i in range(samples_drawn)]
start2 = [Samples_from_start[i][1] for i in range(samples_drawn)]

#Timer stops, for accurate time comment out all the plots
time_taken = time.time() - time_taken
print(f"Time taken = {time_taken}")

#Shows the Gibbs and the new distribution
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

#shows the start distribution (s1, s2) and the resulting distribution (s1, s2|y)
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