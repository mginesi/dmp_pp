import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from dmp import dmp_cartesian as dmp

# This script tests both the original DMP formulation from Ijspeert et al. 2002
# and the new formulation from Park et al. 2008

# Parameters
K = 1000
alpha = 4.0
n_dim = 2
n_bfs = 50
dt = 0.01
tol = 0.05

# DMP initialization
MP_new = dmp.DMPs_cartesian(n_dmps = n_dim, n_bfs = n_bfs, K = K, dt = dt, alpha_s = alpha, tol = tol, rescale=None)
MP_old = dmp.DMPs_cartesian(n_dmps = n_dim, n_bfs = n_bfs, K = K, dt = dt, alpha_s = alpha, tol = tol, rescale='diagonal')

# Trajectory
t = np.linspace(0.0, np.pi, 1000)
x = t
y = np.sin(t) * np.sin(t) + t / 15.0 / np.pi
gamma = np.transpose(np.array([x, y]))
g_old = gamma[-1]

MP_new.imitate_path(x_des = gamma)
MP_old.imitate_path(x_des = gamma)

# New goal positions
g_high = g_old + np.array([0, g_old[-1]])
g_under = g_old * np.array([1, -1])

MP_new.x_goal = g_high
MP_old.x_goal = g_high
mp_new_high = MP_new.rollout()[0]
mp_old_high = MP_old.rollout()[0]

MP_new.x_goal = g_under
MP_old.x_goal = g_under
mp_new_under = MP_new.rollout()[0]
mp_old_under = MP_old.rollout()[0]

plt.figure()
plt.plot(x, y, 'b', label='learned traj.')
plt.plot(mp_new_high[:, 0], mp_new_high[:, 1], '--g', label='Park et al.')
plt.plot(mp_old_high[:, 0], mp_old_high[:, 1], ':k', label='Ijspeert et al.')
plt.plot([-1, np.pi + 1], [0, 0], '-', color = 'gray', label=r'$x_2 = 0$')
plt.legend(loc='best')
plt.plot(gamma[0][0], gamma[0][1], '.k', markersize=10)
plt.plot(g_high[0], g_high[1], '*k', markersize=10)
plt.xlim([-0.2, np.pi + 0.2])
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.figure()
plt.plot(x, y, 'b', label='learned traj.')
plt.plot(mp_new_under[:, 0], mp_new_under[:, 1], '--g', label='Park et al.')
plt.plot(mp_old_under[:, 0], mp_old_under[:, 1], ':k', label='Ijspeert et al.')
plt.plot([-1, np.pi + 1], [0, 0], 'k', color = 'gray', label=r'$x_2 = 0$')
plt.legend(loc='best')
plt.plot(gamma[0][0], gamma[0][1], '.k', markersize=10)
plt.plot(g_under[0], g_under[1], '*k', markersize=10)
plt.xlim([-0.2, np.pi + 0.2])
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.show()