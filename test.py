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
W = em.EStep(mu, Sigma, pi, y)
mu, Sigma, pi = em.MStep(y, W)
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

mu, Sigma, pi = em.EM(2, y)
print(mu)
print(Sigma)
print(pi)

print(em.LogL(mu, Sigma, pi, y))



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





