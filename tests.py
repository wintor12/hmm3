
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import csv
from GMHMM import GMHMM
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import os


n = 9    # states
d = 10   # features
m = 10    # mixtures

def preprocess(path, impute = False, normalize = True):
    f = open(path, 'rb')
    reader = csv.reader(f, delimiter=',')
    headers = reader.next() #skip header    

    dataset = []
    ids = []
    states = []
    if impute is True:
        for row in reader:
            if row[1] not in users.keys():
                users[row[1]] = []
            users[row[1]].append(row)
    else:
        for row in reader:
            if 'nan' not in row:
                dataset.append(row[2:-1])
                ids.append(int(row[1]))
                states.append(int(row[-1]))

    if normalize is True:
        dataset = np.array(dataset, dtype=np.double)
        dataset = preprocessing.scale(dataset)

    feature_selection(dataset, states)
    

    '''
    store dataset by user id and by states respectively
    '''
    users = {}  #key: user id   value:  lists of array  [array(1,2,3), array(1,2,3)]
    users_target = {}
    states_data = {}
    ids = [int(i) for i in ids]
    states = [int(state) for state in states]
    for i in xrange(len(dataset)):
        if ids[i] not in users.keys():
            users[ids[i]] = []
            users_target[ids[i]] = []
        users[ids[i]].append(dataset[i])
        users_target[ids[i]].append(states[i]) 
        if states[i] not in states_data.keys():
            states_data[states[i]] = []
        states_data[states[i]].append(dataset[i])

    print 'Total length: ' + str(len(dataset))
    print '====number of data of each mood====:'
    print states_data.keys()
    print map(len, states_data.values())

    return users, states_data, users_target

def feature_selection(dataset, target):

    X, y = dataset, target
    print X.shape
    clf = ExtraTreesClassifier()
    X_new = clf.fit(X, y).transform(X)
    print 'feature importance: '
    print clf.feature_importances_


def random_init_A():
    '''
    Set A as random valid probability matrices
    '''
    np.random.seed(42)
    A = np.random.random_sample((n, n))
    row_sums = A.sum(axis=1)
    A = np.array(A / row_sums[:, np.newaxis], dtype=np.double)
    return A

def init_others(states_data):
    '''
    Get weights, means and covars through mixture of gaussian on training data
    weights: W  N * M
    means:      N * M * D
    covars:     N * M * (D*D)
    '''
    W = np.zeros(n*m).reshape(n,m)
    means = np.zeros(n*m*d).reshape(n,m,d)
    covars = np.zeros(n*m*d*d).reshape(n,m,d,d)
    # obs = states_data[2]
    # g = gaussMixture(obs)
    # print g.covars_
    # print g.covars_[0]
    for state in states_data.keys():
        print '====process mood===== ' + str(state) 
        obs = np.array(states_data[state], dtype=np.double)
        s = state - 2   #hardcode
        g = gaussMixture(obs)
        for weight in xrange(m):
            W[s][weight] = g.weights_[weight]
            for feature in xrange(d):
                means[s][weight][feature] = g.means_[weight][feature]
                for feature2 in xrange(d):
                    # covars[s][weight][feature][feature2] = g.covars_[weight][feature][feature2]
                    if feature2 == feature:
                        covars[s][weight][feature][feature2] = 0.1
                    else:
                        covars[s][weight][feature][feature2] = 0
    return W, means, covars

def gaussMixture(obs):
    np.random.seed(1)
    g = mixture.GMM(n_components=m, covariance_type='full')
    g.fit(obs)
    return g

def gaussMixtureScore(obs, k):
    '''
    Split dataset to 7:3 as training and testing
    return the sum log probability of test data
    '''
    num_train = int(len(obs)*0.7)
    train = obs[:num_train]
    test = obs[num_train:]
    np.random.seed(1)
    g = mixture.GMM(n_components=k)
    g.fit(train)
    # print np.round(g.weights_, 2)
    # print np.round(g.means_, 2)
    # np.round(g.covars_, 2)
    # print g.get_params()
    return sum(g.score(test))


def bestScoreGM(states_data, max_k):
    '''
    states_data: data stored by state
    max_k: the maximum mixtures of gaussian we try
    try different setting of k(number of mixutres) to find best test result.
    '''
    max_score_k = np.zeros(len(states_data))
    max_score = np.zeros(len(states_data))
    score_mx = []
    for state in states_data.keys():
        for k in xrange(max_k):
            if k == 0:
                score = gaussMixtureScore(np.array(states_data[state],dtype=np.double), 1)
                max_score[state - 2] = score  #Here I hardcode state -2 because state start from 2
                max_score_k[state - 2] = k
                score_mx.append(score)
            else:
                score = gaussMixtureScore(np.array(states_data[state],dtype=np.double), k*5)
                score_mx.append(score)
                if score > max_score[state - 2]:
                    max_score[state - 2] = score
                    max_score_k[state - 2] = k
    print max_score_k
    print max_score
    score_mx = np.array(score_mx).reshape(len(states_data), max_k)
    x = np.arange(0,5*max_k,5)
    for k in xrange(max_k):
        plt.plot(x,score_mx[k])
    plt.show()

def saveResult(path, A, pi, W, means):
    np.savetxt(os.path.join(path, 'A'), A, fmt='%10.3f')
    np.savetxt(os.path.join(path, 'pi'), pi, fmt='%10.3f')
    np.savetxt(os.path.join(path, 'w'), W, fmt='%10.3f')
    means.tofile(os.path.join(path, 'means'), sep=',')


def evaluate(target, predict):
    res = np.zeros(5)
    for i in xrange(len(target)):
        if target[i] - predict[i] == 0:
            res[0] += 1
        if abs(target[i] - predict[i]) <= 1:
            res[1] += 1
        if abs(target[i] - predict[i]) <= 2:
            res[2] += 1
        if abs(target[i] - predict[i]) <= 3:
            res[3] += 1
        if abs(target[i] - predict[i]) <= 4:
            res[4] += 1
    total = len(target)
    return res, total
    # res = res/(np.ones(5)*len(target))
    # print 'exact: ' + str(res[0])
    # print 'diff 1: ' + str(res[1])
    # print 'diff 2: ' + str(res[2])
    # print 'diff 3: ' + str(res[3])
    # print 'diff 4: ' + str(res[4])

def train_hmm_by_user(users, num_users, start_user, A, pi, W, means, covars, iters):
    num_users_train = num_users

    for u in xrange(num_users_train):
        gmmhmm = GMHMM(n,m,d,A,means,covars,W,pi,init_type='user')
        userid = users.keys()[u + start_user]        
        training = users[userid]
        print '===training===' + str(u+start_user) + ' userid ' + str(userid) + ' number of states: ' + str(len(training))
        obs = np.array(training)
        A, pi, W, means, covars = gmmhmm.train(obs,iters)

    return gmmhmm

def test_hmm_by_user(users, users_target, num_users, start_user, gmmhmm):
    total = 0
    res = np.zeros(5)
    percentage = np.zeros(5)
    for u in xrange(num_users):
        userid = users.keys()[u + start_user] 
        test = users[userid]
        target = users_target[userid]   
        predict = gmmhmm.decode(np.array(users[userid]))
        target = users_target[userid] 
        r, t = evaluate(target, predict)
        res = [sum(x) for x in zip(res, r)]
        total += t
    percentage = res/(np.ones(5)*total)
    return res, total, percentage


def test(): 
  
    users, states_data, users_target = preprocess('Project_S_data_DS_intern.csv')
    # bestScoreGM(states_data,9)

    W, means, covars = init_others(states_data)
    A = random_init_A()
    A = [[np.log(c) for c in b] for b in A]
    W = [[np.log(c) for c in b] for b in W]
    ## Init pi based on the count of each state
    pi = np.array(map(len, states_data.values()), dtype=np.double)/sum(map(len, states_data.values()))
    pi = [np.log(b) for b in pi]
    
    gmmhmm = train_hmm_by_user(users, 10, 10, A, pi, W, means, covars, 20)
    
    A = [[np.exp(c) for c in b] for b in gmmhmm.A]
    W = [[np.exp(c) for c in b] for b in gmmhmm.w]
    pi = [np.exp(b)for b in gmmhmm.pi]
    means = gmmhmm.means

    path = os.path.join(os.getcwd(), 'res')
    if not os.path.exists(path):
      os.makedirs(path)
    saveResult(path, A, pi, W, means)

    res, total, percentage = test_hmm_by_user(users, users_target, 5, 20, gmmhmm)
    print res
    print total
    print percentage
    
    # userid = users.keys()[5]
    # print userid

    # # print obs
    # target = users_target[userid]   

    # predict = gmmhmm.decode(np.array(users[userid]))
    # print target
    # print predict

    # evaluate(target, predict)


test()

