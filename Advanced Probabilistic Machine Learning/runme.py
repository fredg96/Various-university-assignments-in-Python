# Runable file for project in Advanced probabilistic Machine Learning
# Plots all graphs from report in order
# Group 7651

from Matchmaking import compare, compareLOL, q8, gibbs_sample, burn_in, adf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

# Hyperparameters
my_s1 = 25
my_s2 = 25
var_s1 = 25/3
var_s2 = 25/3
var_t = 25/3
t_guess = 1
det_T = 1/((1/var_s1+1/var_t)*(1/var_s2+1/var_t)-1/var_t**2)
Sigma_T = det_T*np.array([[1/var_s2 + 1/var_t, 1/var_t], [1/var_t, 1/var_s1 + 1/var_t]])

# Other constants
burn = 200

# Q4 Burn in
s1b, s2b, y_b = burn_in()
plt.plot(y_b, s1b, label='player 1')
plt.plot(y_b, s2b, label='player 2')
plt.xlabel('Num samples')
plt.ylabel('Mean')
plt.title('Burn in')
plt.legend()
plt.show

# Q4 Subplot the 4 different number of samples + time graph
# aware of warnings due to subplot label, but produces correct result
acc_samples = range(1200,5200,1000)
x_acc = np.linspace(0, 50, 1000)
t = []
plt.figure(figsize=(10,10))
for index, sample in enumerate(acc_samples):
    subplot =  plt.subplot(3,2,index+1)
    t_start = time.time()
    s_sample = np.zeros((sample, 2))
    t_sample = np.zeros(sample)
    gibbs_sample(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T, s_sample, t_sample, 1, sample)
    
    s1_spred = np.mean(s_sample[burn:, 0])
    s1_vpred = np.std(s_sample[burn:, 0])
    s2_spred = np.mean(s_sample[burn:, 1])
    s2_vpred = np.std(s_sample[burn:, 1])
    
    subplot.hist(s_sample[burn:], bins=50, density=True)
    subplot.plot(x_acc, sp.stats.norm.pdf(x_acc, s1_spred, s1_vpred), label='p1: m='+"%.3f"%s1_spred+', v='+"%.3f"%s1_vpred)
    subplot.plot(x_acc, sp.stats.norm.pdf(x_acc, s2_spred, s2_vpred), label='p2: m='+"%.3f"%s2_spred+', v='+"%.3f"%s2_vpred)
    subplot.legend(loc='upper right')
    end = time.time()-t_start
    subplot.set_title('N =' + str(sample) + ', Time = ' + "%.2f"%end+" seconds")
    t.append(end)
# Plot computational time as function of samples
plt.subplot(325).plot(acc_samples, t)
plt.subplot(325).set_xlabel('Number of timesteps')
plt.subplot(325).set_ylabel('Time (seconds)')
plt.show()

# Q5 ADF
# If input of adf = 1, matches of seriea is ranomized
adf_mean, adf_std = adf(0)
for team in adf_mean:
    plt.plot(adf_mean[team], label=team)
plt.title('Update of each teams mean value through season, randomized training')
plt.xlabel('Number of matches played')
plt.ylabel('mean')
plt.xlim(right=45)
plt.legend()
plt.show()

# Plot Q8 with 1 and 20 iterations
burn = 20
s_sample1, mulist1 = q8(1)
s1_spred = np.mean(s_sample1[burn:, 0])
s1_vpred = np.std(s_sample1[burn:, 0])
s2_spred = np.mean(s_sample1[burn:, 1])
s2_vpred = np.std(s_sample1[burn:, 1])

x = np.linspace(0, 60, 1000)
plt.hist(s_sample1[200:], bins=50, density=True)
plt.plot(x, sp.stats.norm.pdf(x, s1_spred, s1_vpred),label='P1 GS, m = %1.2f, v= %1.2f' %(s1_spred,s1_vpred))
plt.plot(x, sp.stats.norm.pdf(x, s2_spred, s2_vpred),label='P2 GS, m = %1.2f, v= %1.2f' %(s2_spred,s2_vpred))
plt.plot(x, sp.stats.norm.pdf(x, mulist1[0], mulist1[1]),label='P1 MP, m = %1.2f, v= %1.2f' %(mulist1[0], mulist1[3]))
plt.plot(x, sp.stats.norm.pdf(x, mulist1[2], mulist1[3]),label='P2 MP, m = %1.2f, v= %1.2f' %(mulist1[2], mulist1[3]))
plt.ylabel("Density")
plt.legend()
plt.show()

s_sample20, mulist20 = q8(20)
s1_spred = np.mean(s_sample20[burn:, 0])
s1_vpred = np.std(s_sample20[burn:, 0])
s2_spred = np.mean(s_sample20[burn:, 1])
s2_vpred = np.std(s_sample20[burn:, 1])

x = np.linspace(0, 60, 1000)
plt.hist(s_sample20[200:], bins=50, density=True)
plt.plot(x, sp.stats.norm.pdf(x, s1_spred, s1_vpred),label='P1 GS, m = %1.2f, v= %1.2f' %(s1_spred,s1_vpred))
plt.plot(x, sp.stats.norm.pdf(x, s2_spred, s2_vpred),label='P2 GS, m = %1.2f, v= %1.2f' %(s2_spred,s2_vpred))
plt.plot(x, sp.stats.norm.pdf(x, mulist20[0], mulist20[1]),label='P1 MP, m = %1.2f, v= %1.2f' %(mulist20[0], mulist20[3]))
plt.plot(x, sp.stats.norm.pdf(x, mulist20[2], mulist20[3]),label='P2 MP, m = %1.2f, v= %1.2f' %(mulist20[2], mulist20[3]))
plt.ylabel("Density")
plt.legend()
plt.show()

#Q9 own data from NA League of legends Championship
lol_dicts = compareLOL()
lol_mean = lol_dicts[0]
for team in lol_mean:
    plt.plot(lol_mean[team], label=team)
plt.legend()
plt.xlabel('Number of games played')
plt.ylabel('Mean value')
plt.show()

#Q10 Plot result with draws included
dict_list = compare()
mean_dict = dict_list[0]
for team in mean_dict:
    plt.plot(mean_dict[team], label=team)
plt.xlim(right=55)
plt.legend()
plt.show()
