import numpy as np

import em

# mu, Sigma, pi, y

mu = np.array([[0, 1], [1, 2], [2, 3]])
Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 2]], [[2, 1], [1, 1]]])
pi = np.array([.33, .33])

y = np.array([[0, 1], [1, 2], [2, 3], [1.5, 1.5], [1.5, 2.5], [2.5, 3.5]])

y = np.loadtxt('500202.csv', delimiter=',', skiprows=1)
print(y)
print(y.shape)

# print(mu)
# print(Sigma)
# print(pi)
# print(y)

# print(em.LogL(mu, Sigma, pi, y))
# W = em.EStep(mu, Sigma, pi, y)
# mu, Sigma, pi = em.MStep(y, W)
# print(W)
# print(mu)
# print(Sigma)
# print(pi)

# rng = np.random.default_rng()
# s = np.zeros((3, 2, 2))
# diag_view = np.einsum('...ii->...i',s)
# diag_view[:] = 1
# print(s)
# print(diag_view)

Ks = (2,3,4)
for K in Ks:
    print(K)
    mu, Sigma, pi = em.EM(y, K=K)
    print(f'{mu=}')
    print(f'{Sigma=}')
    print(f'{pi=}')
    print(em.LogL(mu, Sigma, pi, y))
    print(em.BIC(mu, Sigma, pi, y))
    em.plot2D(y, cond=True, pred=(mu, Sigma, pi))

# print(y - mu[:,None])
# print((y[:,1] - mu[:,1][:,None]).shape)

# p = em.predict(y, mu, Sigma, pi, index=1)
# print(p.shape)

# 


# K = len(mu)
# print(mu.size + Sigma.shape[0]*K*(K+1)/2 + len(pi)-1)
# print(P)
# newpi = P.mean(axis=0)
# mean = P.T @ y / P.sum(axis=0)[:,None]
# print(Sigma.shape)
# print(P.T.shape)
# centered = y - mu[:,None]

# print(centered.shape)
# print(centered)
# c2 = np.einsum('ijk,ijl->ijkl', centered, centered)
# print(c2.shape)
# print(c2)
# scaled_c2 = np.einsum('ji,ijkl->ikl', P, c2)

# print(scaled_c2.shape)
# print(scaled_c2)

# print(scaled_c2.shape, P.sum(axis=0)[:,None, None].shape)
# sigma = scaled_c2 / P.sum(axis=0)[:,None, None]

# print(sigma.shape)
# print(sigma)

# k = 3
# i = 3
# p = 5
# #p-i = 2
# mu1=np.ones((k, i))
# mu2=np.ones((k, p-i))
# Sigma11=np.ones((k, i, i))
# Sigma12=np.ones((k, i, p-i))
# Sigma21=np.ones((k, p-i, i))
# Sigma22=np.ones((k, p-i, p-i))
# a_mu1=y[:,:1] - mu1[:,None]


# part1 = Sigma21@Sigma11
# print(part1)
# print(part1.shape)

# part2 = np.einsum('ijk,ilk->lij', part1, a_mu1)
# print(part2)
# print(part2.shape)
# print(W.shape)
# print((W[:,:,None]*part2).sum(axis=1).shape)




