
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    # initial
    mean_1 = 0
    var_1 = 0.7
    mean_2 = 0
    var_2 = 0.7
    beta = 2
    y = 1


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
    x = np.linspace(-5, 5, num=1000)
    plt.plot(x, stats.norm.pdf(x, mean_s1Iy, var_s1Iy))
    plt.plot(x, stats.norm.pdf(x, mean_s2Iy, var_s2Iy))
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