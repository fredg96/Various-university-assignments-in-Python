import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import numpy.random as rng
import math
import pandas as pd
import time

#var_s1 = 25/3
#var_s2 = 25/3
#sigma_s = [[var_s1, 0], [0, var_s2]]

def update_my(t, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T):
    return Sigma_T @ np.array([[my_s1 / var_s1 + t / var_t], [my_s2 / var_s1 - t / var_t]])

    # Gibbs Sampler function
    # s_sample and t_ sample are two/one dimentional vectors storing the values calculated


def gibbs_sample(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T, s_sample, t_sample, y, num_samples):
    sigma_s = [[var_s1, 0], [0, var_s2]]
    my = update_my(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T)
    s_sample[0] = rng.multivariate_normal(my.T[0], sigma_s)
    t_sample[0] = t_guess
    a = -math.inf
    b = 0
    if (y > 0):
        a = 0
        b = math.inf
    for i in range(num_samples):
        # my_t = Sigma_T@np.array([[my_s1/var_s1 + t_sample[i-1]/var_t], [my_s2/var_s1 - t_sample[i-1]/var_t]])
        my = update_my(t_sample[i - 1], my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T)
        s_sample[i] = rng.multivariate_normal(my.T[0], Sigma_T)
        t_sample[i] = sp.stats.truncnorm.rvs(a, b, s_sample[i][0] - s_sample[i][1], var_t)

def burn_in():
    sample = 1000
    my_s1 = 25
    my_s2 = 25
    var_s1 = 25/3
    var_s2 = 25/3
    var_t = 25/3
    t_guess = 10
    det_T = 1/((1/var_s1+1/var_t)*(1/var_s2+1/var_t)-1/var_t**2)
    Sigma_T = det_T*np.array([[1/var_s2 + 1/var_t, 1/var_t], [1/var_t, 1/var_s1 + 1/var_t]])
    s_sample = np.zeros((sample, 2))
    t_sample = np.zeros(sample)
    gibbs_sample(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T, s_sample, t_sample, 1, sample)
        
    # Burn in
    x = np.linspace(0, 60, 1000)
    s1b = []
    s2b = []
    y = range(10, 1000, 10)
    for burn in y:
        s1_spred = np.mean(s_sample[:burn, 0])
        s1_vpred = np.std(s_sample[:burn, 0])
        s2_spred = np.mean(s_sample[:burn, 1])
        s2_vpred = np.std(s_sample[:burn, 1])
        s1b.append(s1_spred)
        s2b.append(s2_spred)
    return(s1b, s2b, y)

def truncGaussMM(my_a, my_b, m1, s1):
    a, b = (my_a - m1) / np.sqrt(s1), (my_b - m1) / np.sqrt(s1)
    m = sp.stats.truncnorm.mean(a, b, loc=m1, scale=np.sqrt(s1))
    s = sp.stats.truncnorm.var(a, b, loc=0, scale=np.sqrt(s1))
    return m, s

def mutiplyGauss(m1, s1, m2, s2):
    s = s1*s2/(s1+s2)
    m = (m1*s2 + m2*s1)/(s1+s2)
    return m, s

def divideGauss(m1, s1, m2, s2):
    m = (m1*s2 - m2*s1)/(s2 - s1)
    s = s1*s2/(s2-s1)

    return m, s

def MP(m1,s1,m2,s2,st,y):
    mu_s1_m = m1
    mu_s1_s = s1
    mu_s2_m = m2
    mu_s2_s = s2
    for i in range(100):
        # message mu1 From s1 to factor f3
        mu_s1_f3_m = mu_s1_m
        mu_s1_f3_s = mu_s1_s
        #message mu2 from s2 to factor f3
        mu_s2_f3_m = mu_s2_m
        mu_s2_f3_s = mu_s2_s
        #from factor f3 to w
        mu_f3_w_m = mu_s1_f3_m - mu_s2_f3_m
        mu_f3_w_s = mu_s1_f3_s + mu_s2_f3_s
        #from w to factor f4
        mu_w_f4_m = mu_f3_w_m
        mu_w_f4_s = mu_f3_w_s
        #from factor f4 to t
        mu_f4_t_m = mu_w_f4_m
        mu_f4_t_s = mu_w_f4_s+st
        #Moment matching marginal of t
        if y == 1:
            a,b = 0,100000000
        else:
            a,b = -100000000,0
        pt_m, pt_s = truncGaussMM(a,b,mu_f4_t_m, mu_f4_t_s )
        #from factor f5 to y
        mu_f5_y_m, mu_f5_y_s = divideGauss(pt_m ,pt_s ,mu_f4_t_m, mu_f4_t_s )
        #from t to f4
        mu_t_f4_m = mu_f5_y_m
        mu_t_f4_s = mu_f5_y_s
        #from f4 to w
        mu_f4_w_m = mu_t_f4_m
        mu_f4_w_s = mu_t_f4_s+st
        #from w to f3
        mu_w_f3_m = mu_f4_w_m
        mu_w_f3_s = mu_f4_w_s
        #from f3 to s1
        mu_f3_s1_m = mu_s2_f3_m+mu_w_f3_m
        mu_f3_s1_s = mu_w_f3_s+mu_s2_f3_s
        #from f3 to s2
        mu_f3_s2_m = mu_s1_f3_m-mu_w_f3_m
        mu_f3_s2_s = mu_w_f3_s+mu_s1_f3_s

        # Update s1

        mu_s1_m, mu_s1_s = mutiplyGauss(mu_s1_m ,mu_s1_s,mu_f3_s1_m, mu_f3_s1_s)

        #Update s2

        mu_s2_m, mu_s2_s = mutiplyGauss(mu_s2_m,mu_s2_s,mu_f3_s2_m, mu_f3_s2_s)

    return mu_s1_m, mu_s1_s, mu_s2_m, mu_s2_s

def simplePred(m1,m2,s1,s2):
    tol = 2
    if np.abs(m1 - m2) < tol:
        p1 = 0
        p2 = 0
        for i in range(5):
            res1 = np.random.normal(m1, s1, 1)
            res2 = np.random.normal(m2, s2, 1)
            if np.sign(res1 - res2) == 1:
                p1 += 1
            else:
                p2 += 1
        res = p1 - p2
    else:
        res = m1 - m2
    return res

def advPred(m1,m2,s1,s2,tol):
    f = 1/(1+np.exp(-(m1/s1-m2/s2)))
    if f>tol:
        res = 1
    else:
        res = -1
    return res

def predAll(m1,m2,s1,s2,tol,equ):
    f = 1/(1+np.exp(-(m1/s1-m2/s2)))
    if np.abs(f)<equ:
        res = 0
    else:
        if f > tol:
            res = 1
        else:
            res = -1
    return res


def q8(iterations):
    m1 = 25
    m2 = 25
    s1 = 25/3
    s2 = 25/3
    t = 25/3
    mu_s1_m = m1
    mu_s1_s = s1
    mu_s2_m = m2
    mu_s2_s = s2
    st = t
    y = 1
    N = 2200
    burn = 200
    t_g = 2
    det = 1 / ((1 / s1 + 1 / st) * (1 / s2 + 1 / st) - 1 / st ** 2)
    cov = det * np.array([[1 / s2 + 1 / st, 1 / st], [1 / st, 1 / s2 + 1 / st]])
    s_sample1 = np.zeros((N, 2))
    t_sample1 = np.zeros(N)

    # Message passing
    for i in range(iterations):
        # message mu1 From s1 to factor f3
        mu_s1_f3_m = mu_s1_m
        mu_s1_f3_s = mu_s1_s
        #message mu2 from s2 to factor f3
        mu_s2_f3_m = mu_s2_m
        mu_s2_f3_s = mu_s2_s
        #from factor f3 to w
        mu_f3_w_m = mu_s1_f3_m - mu_s2_f3_m
        mu_f3_w_s = mu_s1_f3_s + mu_s2_f3_s
        #from w to factor f4
        mu_w_f4_m = mu_f3_w_m
        mu_w_f4_s = mu_f3_w_s
        #from factor f4 to t
        mu_f4_t_m = mu_w_f4_m
        mu_f4_t_s = mu_w_f4_s+st
        #Moment matching marginal of t
        if y == 1:
            a,b = 0,100000000
        else:
            a,b = -100000000,0
        pt_m, pt_s = truncGaussMM(a,b,mu_f4_t_m, mu_f4_t_s )
        #from factor f5 to y
        mu_f5_y_m, mu_f5_y_s = divideGauss(pt_m ,pt_s ,mu_f4_t_m, mu_f4_t_s )
        #from t to f4
        mu_t_f4_m = mu_f5_y_m
        mu_t_f4_s = mu_f5_y_s
        #from f4 to w
        mu_f4_w_m = mu_t_f4_m
        mu_f4_w_s = mu_t_f4_s+st
        #from w to f3
        mu_w_f3_m = mu_f4_w_m
        mu_w_f3_s = mu_f4_w_s
        #from f3 to s1
        mu_f3_s1_m = mu_s2_f3_m+mu_w_f3_m
        mu_f3_s1_s = mu_w_f3_s+mu_s2_f3_s
        #from f3 to s2
        mu_f3_s2_m = mu_s1_f3_m-mu_w_f3_m
        mu_f3_s2_s = mu_w_f3_s+mu_s1_f3_s

        # Update s1

        mu_s1_m, mu_s1_s = mutiplyGauss(mu_s1_m ,mu_s1_s,mu_f3_s1_m, mu_f3_s1_s)

        #Update s2

        mu_s2_m, mu_s2_s = mutiplyGauss(mu_s2_m,mu_s2_s,mu_f3_s2_m, mu_f3_s2_s)
        mus = [mu_s1_m, mu_s1_s, mu_s2_m, mu_s2_s]
    gibbs_sample(t_g, m1, s1, m2, s2, st, cov, s_sample1, t_sample1, y, N)
    return(s_sample1, mus)

def adf(rand):
    # Q5 & Q6 Serie A ADF and prediction
    #####################################
    t_start = time.time()
    with open('SerieA.CSV', 'r') as f:
        A = pd.read_csv(f)
    print(A)

    mean_dict = dict()
    std_dict = dict()
    r_dict = dict()
    i_dict = dict()
    start_mean = 25
    start_std = 25/3
    N = 2200
    # Fill dicts with team name and init values
    for team in A['team1']:
        if(team not in mean_dict):
            mean_dict[team] = [start_mean]
            std_dict[team] = [start_std]
            r_dict[team] = []
            i_dict[team] = []      
    vart = 25/3
    t_g = 5
    guess_correct = 0
    guess_total = 0
    draw = 0
    A_rand = A.sample(frac=1).reset_index(drop=True)
    #for index, row in A_rand.iterrows():
    if(rand == 1):
        use = A_rand
    else:
        use = A
    for index, row in use.iterrows():
        if(index%50 == 0):
            print(index)
        
        result = row['score1']-row['score2']
        r_dict[row['team1']].append(result)
        r_dict[row['team2']].append(-result)
        i_dict[row['team1']].append(index)
        i_dict[row['team2']].append(index)
        if(result==0):
            draw +=1      
        else:
            s_sample1 = np.zeros((N, 2))
            t_sample1 = np.zeros(N)
            my1 = mean_dict[row['team1']][-1]
            my2 = mean_dict[row['team2']][-1]
            std1 = std_dict[row['team1']][-1]
            std2 = std_dict[row['team2']][-1]
            my = [my1, my2]
            var = [[std1, 0], [0, std2]]
            
            # Predict result
            pred = simplePred(my1, my2, std1, std2)
            if((pred >0 and result>0) or (pred<0 and result<0)):
                guess_correct+=1
            guess_total+=1
            
            det = 1/((1/std1+1/vart)*(1/std2+1/vart)-1/vart**2)
            cov = det*np.array([[1/std2 + 1/vart, 1/vart], [1/vart, 1/std2 + 1/vart]])
            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample1, t_sample1, result, N)
            
            b_in = 400

            # Calculate new mean and var, disregard burn-in
            s1spred = np.mean(s_sample1[b_in:, 0])
            s1vpred = np.std(s_sample1[b_in:, 0])
            s2spred = np.mean(s_sample1[b_in:, 1])
            s2vpred = np.std(s_sample1[b_in:, 1])
            # Insert new values in dict
            mean_dict[row['team1']].append(s1spred)
            std_dict[row['team1']].append(s1vpred)
            mean_dict[row['team2']].append(s2spred)
            std_dict[row['team2']].append(s2vpred)
            
    print(draw)
    print('total time :',time.time()-t_start)
    return(mean_dict, std_dict)

def compare():
    my_s1 = 25
    my_s2 = 25
    var1 = 25/3
    var2 = 25/3
    vart = 25 / 3
    t_g = 1
    tol = 0.5
    tol2 = tol
    drawMargin = 0.1
    start_mean = my_s1
    start_std = var1

    correct = 0
    correct_MP = 0
    correctDraw = 0
    correctSimpleDraw = 0
    correctSimple = 0

    corr_rand = 0
    nr_guess = 0
    nr_guessDraws = 0

    N = 2200
    b_in = 200

    with open('SerieA.CSV', 'r') as f:
        A = pd.read_csv(f)

    #A = A.sample(frac=1).reset_index(drop=True)
    mean_dict = dict()
    std_dict = dict()
    mean_dict_MP = dict()
    std_dict_MP = dict()

    # Fill dicts with team name and init values
    for team in A['team1']:
        if (team not in mean_dict):
            mean_dict[team] = [start_mean]
            std_dict[team] = [start_std]
            mean_dict_MP[team] = [start_mean]
            std_dict_MP[team] = [start_std]

    for index, row in A.iterrows():
        my1 = mean_dict[row['team1']][-1]
        my2 = mean_dict[row['team2']][-1]
        std1 = std_dict[row['team1']][-1]
        std2 = std_dict[row['team2']][-1]

        result = row['score1'] - row['score2']
        resDraw = predAll(my1,my2,std1,std2,tol,drawMargin)
        resAdv = advPred(my1,my2,std1,std2,tol2)
        res = simplePred(my1,my2,std1,std2)


        if (result == 0):
            print('draw')

            s_sample1 = np.zeros((N, 2))
            t_sample1 = np.zeros(N)
            s_sample2= np.zeros((N, 2))
            t_sample2 = np.zeros(N)
            det = 1 / ((1 / std1 + 1 / vart) * (1 / std2 + 1 / vart) - 1 / vart ** 2)
            cov = det * np.array([[1 / std2 + 1 / vart, 1 / vart], [1 / vart, 1 / std2 + 1 / vart]])
            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample1, t_sample1, 1, N)
            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample2, t_sample2, -1, N)

            s1spred = np.mean(s_sample1[b_in:, 0])
            s1vpred = np.std(s_sample1[b_in:, 0])
            s2spred = np.mean(s_sample1[b_in:, 1])
            s2vpred = np.std(s_sample1[b_in:, 1])

            s1spred2 = np.mean(s_sample2[b_in:, 0])
            s1vpred2 = np.std(s_sample2[b_in:, 0])
            s2spred2 = np.mean(s_sample2[b_in:, 1])
            s2vpred2 = np.std(s_sample2[b_in:, 1])

            s1M = (s1spred+s1spred2)/2
            s1S = (s1vpred+s1vpred2)/2
            s2M = (s2spred + s2spred2) / 2
            s2S = (s2vpred + s2vpred2) / 2

            mean_dict[row['team1']].append(s1M)
            std_dict[row['team1']].append(s1S)
            mean_dict[row['team2']].append(s2M)
            std_dict[row['team2']].append(s2S)
            if resDraw == 0:
                correctDraw += 1
            else:
                drawMargin = 0.01*(1-drawMargin)+drawMargin
            nr_guessDraws += 1


        else:
            s_sample1 = np.zeros((N, 2))
            t_sample1 = np.zeros(N)

            my1_MP = mean_dict_MP[row['team1']][-1]
            my2_MP = mean_dict_MP[row['team2']][-1]
            std1_MP = std_dict_MP[row['team1']][-1]
            std2_MP = std_dict_MP[row['team2']][-1]

            res_MP = simplePred(my1_MP,my2_MP,std1_MP,std2_MP)

            corrRandS1 = np.random.normal(25, 5, 1)
            corrRandS2 = np.random.normal(25, 5, 1)
            nr_guess += 1
            nr_guessDraws += 1

            det = 1 / ((1 / std1 + 1 / vart) * (1 / std2 + 1 / vart) - 1 / vart ** 2)
            cov = det * np.array([[1 / std2 + 1 / vart, 1 / vart], [1 / vart, 1 / std2 + 1 / vart]])

            if np.sign(resDraw) - np.sign(result) == 0:
                correctDraw += 1
            else:
                if np.sign(resDraw)>0:
                    tol = 0.1*(1-tol)+tol
                else:
                    tol = tol-0.1*tol

            if np.sign(resAdv) - np.sign(result) == 0:
                correct += 1
            else:
                if np.sign(resAdv)>0:
                    tol2 = 0.1*(1-tol2)+tol2
                else:
                    tol2 = tol2-0.1*tol2

            if np.sign(res_MP) - np.sign(result) == 0:
                correct_MP += 1

            if np.sign(res) - np.sign(result) == 0:
                correctSimpleDraw += 1
                correctSimple += 1

            if np.sign(corrRandS1 - corrRandS2) - np.sign(result) == 0:
                corr_rand += 1

            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample1, t_sample1, result, N)

            # Calculate new mean and var
            s1spred = np.mean(s_sample1[b_in:, 0])
            s1vpred = np.std(s_sample1[b_in:, 0])
            s2spred = np.mean(s_sample1[b_in:, 1])
            s2vpred = np.std(s_sample1[b_in:, 1])
            # Insert new values in dict
            mean_dict[row['team1']].append(s1spred)
            std_dict[row['team1']].append(s1vpred)
            mean_dict[row['team2']].append(s2spred)
            std_dict[row['team2']].append(s2vpred)

            my1_MP,std1_MP,my2_MP,std2_MP = MP(my1_MP,std1_MP,my2_MP,std2_MP,vart,np.sign(result))
            mean_dict_MP[row['team1']].append(my1_MP)
            std_dict_MP[row['team1']].append(std1_MP)
            mean_dict_MP[row['team2']].append(my2_MP)
            std_dict_MP[row['team2']].append(std2_MP)
    

    print("Prediction rate Gibbs, moddeling draws %f" %(correctDraw/nr_guessDraws))
    print("Prediction rate Gibbs %f" %(correct/nr_guess))
    print("Prediction rate random %f" %(corr_rand / nr_guess))
    print("Prediction rate MP %f" %(correct_MP/nr_guess))
    print("Prediction rate simple %f" % (correctSimple / nr_guess))
    return(mean_dict, std_dict, mean_dict_MP, std_dict_MP)


def compareLOL():
    my_s1 = 25
    my_s2 = 25
    var1 = 25/3
    var2 = 25/3
    vart = 25 / 3
    t_g = 1
    tol = 0.5
    tol2 = tol
    drawMargin = 0.1
    start_mean = my_s1
    start_std = var1

    correct = 0
    correct_MP = 0
    correctDraw = 0
    correctSimpleDraw = 0
    correctSimple = 0

    corr_rand = 0
    nr_guess = 0
    nr_guessDraws = 0

    N = 2200
    b_in = 200

    with open('LeagueOfLegends.CSV', 'r') as f:
        A = pd.read_csv(f)

    #A = A.sample(frac=1).reset_index(drop=True)
    mean_dict = dict()
    std_dict = dict()
    mean_dict_MP = dict()
    std_dict_MP = dict()

    # Fill dicts with team name and init values
    for team in A['blueTeamTag'][610:1114]:
        if (team not in mean_dict):
            mean_dict[team] = [start_mean]
            std_dict[team] = [start_std]
            mean_dict_MP[team] = [start_mean]
            std_dict_MP[team] = [start_std]
    print(A[610:1114])
    for index, row in A[610:1114].iterrows():
        if(index%100==0):
            print(index)
        my1 = mean_dict[row['blueTeamTag']][-1]
        my2 = mean_dict[row['redTeamTag']][-1]
        std1 = std_dict[row['blueTeamTag']][-1]
        std2 = std_dict[row['redTeamTag']][-1]

        result = row['bResult'] - row['rResult']
        resDraw = predAll(my1,my2,std1,std2,tol,drawMargin)
        resAdv = advPred(my1,my2,std1,std2,tol2)
        res = simplePred(my1,my2,std1,std2)


        if (result == 0):
            print('draw')

            s_sample1 = np.zeros((N, 2))
            t_sample1 = np.zeros(N)
            s_sample2= np.zeros((N, 2))
            t_sample2 = np.zeros(N)
            det = 1 / ((1 / std1 + 1 / vart) * (1 / std2 + 1 / vart) - 1 / vart ** 2)
            cov = det * np.array([[1 / std2 + 1 / vart, 1 / vart], [1 / vart, 1 / std2 + 1 / vart]])
            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample1, t_sample1, 1, N)
            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample2, t_sample2, -1, N)

            s1spred = np.mean(s_sample1[b_in:, 0])
            s1vpred = np.std(s_sample1[b_in:, 0])
            s2spred = np.mean(s_sample1[b_in:, 1])
            s2vpred = np.std(s_sample1[b_in:, 1])

            s1spred2 = np.mean(s_sample2[b_in:, 0])
            s1vpred2 = np.std(s_sample2[b_in:, 0])
            s2spred2 = np.mean(s_sample2[b_in:, 1])
            s2vpred2 = np.std(s_sample2[b_in:, 1])

            s1M = (s1spred+s1spred2)/2
            s1S = (s1vpred+s1vpred2)/2
            s2M = (s2spred + s2spred2) / 2
            s2S = (s2vpred + s2vpred2) / 2

            mean_dict[row['team1']].append(s1M)
            std_dict[row['team1']].append(s1S)
            mean_dict[row['team2']].append(s2M)
            std_dict[row['team2']].append(s2S)
            if resDraw == 0:
                correctDraw += 1
            else:
                drawMargin = 0.01*(1-drawMargin)+drawMargin
            nr_guessDraws += 1


        else:
            s_sample1 = np.zeros((N, 2))
            t_sample1 = np.zeros(N)

            my1_MP = mean_dict_MP[row['blueTeamTag']][-1]
            my2_MP = mean_dict_MP[row['redTeamTag']][-1]
            std1_MP = std_dict_MP[row['blueTeamTag']][-1]
            std2_MP = std_dict_MP[row['redTeamTag']][-1]

            res_MP = simplePred(my1_MP,my2_MP,std1_MP,std2_MP)

            corrRandS1 = np.random.normal(25, 5, 1)
            corrRandS2 = np.random.normal(25, 5, 1)
            nr_guess += 1
            nr_guessDraws += 1

            det = 1 / ((1 / std1 + 1 / vart) * (1 / std2 + 1 / vart) - 1 / vart ** 2)
            cov = det * np.array([[1 / std2 + 1 / vart, 1 / vart], [1 / vart, 1 / std2 + 1 / vart]])

            if np.sign(resDraw) - np.sign(result) == 0:
                correctDraw += 1
            else:
                if np.sign(resDraw)>0:
                    tol = 0.1*(1-tol)+tol
                else:
                    tol = tol-0.1*tol

            if np.sign(resAdv) - np.sign(result) == 0:
                correct += 1
            else:
                if np.sign(resAdv)>0:
                    tol2 = 0.1*(1-tol2)+tol2
                else:
                    tol2 = tol2-0.1*tol2

            if np.sign(res_MP) - np.sign(result) == 0:
                correct_MP += 1

            if np.sign(res) - np.sign(result) == 0:
                correctSimpleDraw += 1
                correctSimple += 1

            if np.sign(corrRandS1 - corrRandS2) - np.sign(result) == 0:
                corr_rand += 1

            gibbs_sample(t_g, my1, std1, my2, std2, vart, cov, s_sample1, t_sample1, result, N)

            # Calculate new mean and var
            s1spred = np.mean(s_sample1[b_in:, 0])
            s1vpred = np.std(s_sample1[b_in:, 0])
            s2spred = np.mean(s_sample1[b_in:, 1])
            s2vpred = np.std(s_sample1[b_in:, 1])
            # Insert new values in dict
            mean_dict[row['blueTeamTag']].append(s1spred)
            std_dict[row['blueTeamTag']].append(s1vpred)
            mean_dict[row['redTeamTag']].append(s2spred)
            std_dict[row['redTeamTag']].append(s2vpred)

            my1_MP,std1_MP,my2_MP,std2_MP = MP(my1_MP,std1_MP,my2_MP,std2_MP,vart,np.sign(result))
            mean_dict_MP[row['blueTeamTag']].append(my1_MP)
            std_dict_MP[row['blueTeamTag']].append(std1_MP)
            mean_dict_MP[row['redTeamTag']].append(my2_MP)
            std_dict_MP[row['redTeamTag']].append(std2_MP)
    print("Prediction rate Gibbs, moddeling draws %f" %(correctDraw/nr_guessDraws))
    print("Prediction rate Gibbs %f" %(correct/nr_guess))
    print("Prediction rate random %f" %(corr_rand / nr_guess))
    print("Prediction rate MP %f" %(correct_MP/nr_guess))
    print("Prediction rate simple %f" % (correctSimple / nr_guess))
    
    return(mean_dict, std_dict, mean_dict_MP, std_dict_MP)
