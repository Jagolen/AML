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
    print(mu_s_vector)
    return np.random.multivariate_normal(mu_s_transpose, std_s_matrix)


mu_s = 1
std_s = 4
mu_s_vector = np.array([[mu_s], [mu_s]])
std_s_matrix = np.array([[std_s**2, 0], [0, std_s**2]])
A = np.array([[1, -1]])
std_t = 3

Iters = 100
All_S = np.zeros((Iters, 2))
All_t = np.zeros(Iters)

All_S[0] = [np.random.normal(loc=mu_s, scale = std_s), np.random.normal(loc=mu_s, scale = std_s)]

for i in range(Iters-1):
    All_t[i] = s_to_t(All_S[0,0], All_S[0,1], std_t)
    All_S[i+1] = t_to_s(All_t[i], A)

x = [i for i in range(Iters)]
y1 = [All_S[i][0] for i in range(Iters)]
y2 = [All_S[i][1] for i in range(Iters)]

plt.plot(x,y1)
plt.plot(x,y2)
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