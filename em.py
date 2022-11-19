import numpy as np
import scipy.stats as stats

def LogL(mu, Sigma, pi, y):
    if len(pi) > len(mu) or len(pi) > len(Sigma):
        Exception("There are too many segments")
    if len(pi) + 1 == len(mu) or len(pi) + 1 == len(Sigma):
        pi = np.append(pi, 1-sum(pi))

    mu = mu[:len(pi)]
    Sigma = Sigma[:len(pi)]

    ll = 0
    for i in y:
        lik = 0
        # print(stats.multivariate_normal.pdf([i for _ in pi], mean=mu, cov=Sigma))
        for j, p in enumerate(pi):
            lik +=  p*stats.multivariate_normal.pdf(i, mean=mu[j], cov=Sigma[j])
        ll += np.log(lik)
    return ll

def BIC(mu, Sigma, pi, y):
    K = len(mu)
    k = mu.size + Sigma.shape[0]*K*(K+1)/2 + len(pi)-1
    N = len(y)
    ll = LogL(mu, Sigma, pi, y)
    
    bic = k * np.log(N) - 2*ll
    return bic

def EStep(mu, Sigma, pi, y):
    if len(pi) > len(mu) or len(pi) > len(Sigma):
        raise Exception("There are too many segments")
    if len(pi) + 1 == len(mu) or len(pi) + 1 == len(Sigma):
        pi = np.append(pi, 1-sum(pi))
    N = len(y)
    k = len(pi)

    W = np.zeros((N, k))
    for i, yi in enumerate(y):
        for j, p in enumerate(pi):
            W[i, j] = p*stats.multivariate_normal.pdf(yi, mean=mu[j], cov=Sigma[j])

    W /= W.sum(axis=1)[:,None]
    return W

def MStep(y, W):
    mu = W.T @ y / W.sum(axis=0)[:,None]
    
    center = y - mu[:,None]
    c2 = np.einsum('ijk,ijl->ijkl', center, center)
    weighted_c2 = np.einsum('ji,ijkl->ikl', W, c2)
    Sigma = weighted_c2 / W.sum(axis=0)[:, None, None]

    pi = W.mean(axis=0)
    
    return mu, Sigma, pi

def EM(K, y, eps=1e-5):
    rng = np.random.default_rng()
    size = y.shape[1]
    N = len(y)

    mu_i = rng.choice(N, size=K, replace=False)
    mu = y[mu_i]

    Sigma = np.zeros((K, size, size))
    diag_view = np.einsum('...ii->...i', Sigma)
    diag_view[:] = 1

    pi = rng.dirichlet(np.ones(K), size=1).squeeze()

    W = np.zeros((N, K))


    for _ in range(500):
        W = EStep(mu, Sigma, pi, y)
        mu, Sigma, pi = MStep(y, W)
    
    return mu, Sigma, pi


