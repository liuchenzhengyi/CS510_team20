import numpy as np
from scipy.special import logsumexp

def log_ot(C, mu, epi, alpha, inIter, outIter):
    K = len(mu)
    Q = - C / epi
    N = []
    u = []
    s = (np.ones((K,K)) - 2 * np.eye(K)).astype(np.int32).tolist()
    for i in range(K):
        mu[i] = np.log(mu[i])
        N.append(len(mu[i]))
        u.append(np.log(np.ones(N[i])/N[i]))
    for i in range(outIter):
        P = []
        U = np.zeros(N)
        for j in range(K):
            U = U + np.reshape(u[j],s[j])
        logT = Q + U
        for j in range(K):
            ax = list(range(j)) + list(range(j+1,K))
            P.append(logsumexp(logT, axis = tuple(ax)))
        for j in range(inIter):
            for k in range(K):
                u[k] = alpha * u[k] + (1 - alpha) * (u[k] + mu[k] - P[k])
    T = np.exp(logT)
    return T
