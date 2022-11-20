import matplotlib.pyplot as plt
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
            try:
                lik +=  p*stats.multivariate_normal.pdf(i, mean=mu[j], cov=Sigma[j])
            except np.linalg.LinAlgError: # Not Positive Semi Definite
                return -np.inf
            
        ll += np.log(lik)
    return ll

def BIC(mu, Sigma, pi, y):
    K = len(mu)
    k = mu.size + Sigma.shape[0]*K*(K+1)/2 + len(pi)-1
    N = len(y)
    ll = LogL(mu, Sigma, pi, y)
    
    bic = k * np.log(N) - 2*ll
    return bic

def generate_params(rng, y, K):
    size = y.shape[1]
    N = len(y)

    mu_i = rng.choice(N, size=K, replace=False)
    mu = y[mu_i]

    Sigma = np.zeros((K, size, size))
    diag_view = np.einsum('...ii->...i', Sigma)
    diag_view[:] = 1

    pi = rng.dirichlet(np.ones(K), size=1).squeeze()

    return mu, Sigma, pi

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
            try:
                W[i, j] = p*stats.multivariate_normal.pdf(yi, mean=mu[j], cov=Sigma[j])
            except np.linalg.LinAlgError: # Not Positive Semi Definite
                W[i, j] = 0

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

def EM(y, K, iter=10, eps=1e-4, seed=0, diff='relative'):
    rng = np.random.default_rng(seed=seed)

    if diff == 'simple':
        diff_func = simple_diff
    elif diff == 'relative':
        diff_func = relative_diff
    else:
        raise Exception(f"This difference function: {diff=} is not defined")

    best_ll = -np.inf

    for _ in range(iter):
        mu, Sigma, pi = generate_params(rng, y, K)

        ll_prev, ll = -np.inf, LogL(mu, Sigma, pi, y)

        while diff_func(ll_prev, ll) > eps:
            W = EStep(mu, Sigma, pi, y)
            mu, Sigma, pi = MStep(y, W)

            ll_prev, ll = ll, LogL(mu, Sigma, pi, y)
        
        if ll > best_ll:
            best_ll = ll
            best_mu, best_Sigma, best_pi = mu, Sigma, pi
    
    return best_mu, best_Sigma, best_pi

def predict(y, mu, Sigma, pi, index=1):
    W = EStep(mu, Sigma, pi, y)

    mu1 = mu[:, :index]
    # mu2 = mu[:, index:]
    Sigma11 = Sigma[:, :index, :index]
    # Sigma12 = Sigma[:, :index, index:]
    Sigma21 = Sigma[:, index:, :index]
    # Sigma22 = Sigma[:, index:, index:]

    part1 = Sigma21@np.linalg.inv(Sigma11)
    part2 = y[:,:1] - mu1[:,None]
    new_mu = np.einsum('ijk,ilk->lij', part1, part2)

    p = (W[:,:,None]*new_mu).sum(axis=1)
    return p

def plot2D(y, cond=False, **kwargs):
    if y.ndim != 2:
        raise Exception("Y does not meet the dimension requirements")

    plt.scatter(y[:, 0], y[:, 1])

    if cond is True:
        x = np.linspace(y[:, 0].min(), y[:, 0].max(), 500)
        x = np.vstack((x,x)).T

        p = predict(x, *kwargs['pred'])
        plt.plot(x, p)

    plt.show()

def simple_diff(v1, v2):
    return v2-v1

def relative_diff(v1, v2):
    if any(np.isneginf([v1, v2])):
        return np.inf
    return (v1-v2)/v1