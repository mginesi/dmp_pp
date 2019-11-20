import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import tqdm

t_span = np.linspace(0.0, 1.0, 1000)
T = t_span[-1] - t_span[0]
alpha = 5.0
s_span = np.exp(-alpha * t_span)
traj = (- 4.0 * np.abs(t_span - 0.5) + 1.0) * (t_span > 1/4) * (t_span < 3/4)
K = 100
range_nbfs = (np.ceil(np.power(2, np.linspace(1,7,10)))).astype(int) # basis functions range
err_gaus = np.zeros(len(range_nbfs))
err_moll = np.zeros(len(range_nbfs))
err_wen_2 = np.zeros(len(range_nbfs))
err_wen_4 = np.zeros(len(range_nbfs))
err_wen_6 = np.zeros(len(range_nbfs))
err_wen_8 = np.zeros(len(range_nbfs))
err_wen_3 = np.zeros(len(range_nbfs))
err_wen_5 = np.zeros(len(range_nbfs))
err_wen_7 = np.zeros(len(range_nbfs))
err_t_g = np.zeros(len(range_nbfs))
err_t_gaus_biased = np.zeros(len(range_nbfs))
count = -1
for n_bfs in tqdm.tqdm(range_nbfs):
    count += 1
    psi_set = np.zeros([n_bfs + 1, 1000])
    truncated_psi_set = np.zeros([n_bfs + 1, 1000])
    phi_set = np.zeros([n_bfs + 1, 1000])
    wen_set_2 = np.zeros([n_bfs + 1, 1000])
    wen_set_4 = np.zeros([n_bfs + 1, 1000])
    wen_set_6 = np.zeros([n_bfs + 1, 1000])
    wen_set_8 = np.zeros([n_bfs + 1, 1000])
    wen_set_3 = np.zeros([n_bfs + 1, 1000])
    wen_set_5 = np.zeros([n_bfs + 1, 1000])
    wen_set_7 = np.zeros([n_bfs + 1, 1000])
    c_i = np.exp(-alpha * np.arange(0, n_bfs + 1, 1) * T / n_bfs) # centers
    h_i = (np.diff(c_i)) ** (-2.0)
    h_i = np.append(h_i, h_i[-1]) # widths for gaussian
    a_i = (np.diff(c_i)) ** (-1.0)
    a_i = np.append(a_i[0], a_i) # widths for mollifiers
    k_i = (np.diff(c_i)) ** (-1.0)
    k_i = np.append(k_i[0], k_i) # widths for wendland
    for it in range(n_bfs+1):
        psi_set[it] = np.exp(-h_i[it] * (s_span - c_i[it]) ** 2)
        truncated_psi_set[it] = np.exp(- 0.5 * h_i[it] * (s_span - c_i[it]) ** 2) * ((s_span - c_i[it]) <= (K * h_i[it] ** (-0.5)))
        xi_tmp = a_i[it] * (s_span - c_i[it])
        phi_set[it] = np.exp(-1.0 / (1.0 - xi_tmp ** 2)) * (np.abs(xi_tmp) < 1.0)
        r_tmp = - np.abs(s_span - c_i[it]) * k_i[it] # the minus balance the fact that the centers are ordered from the greater to the lower, causing k_i < 0
        wen_set_2[it] = (1 - r_tmp) ** 2 * (r_tmp < 1.0) # wenland set order 0
        wen_set_4[it] = (1 - r_tmp) ** 4 * (4 * r_tmp + 1) * (r_tmp < 1.0) # wenland set order 2
        wen_set_6[it] = (1 - r_tmp) ** 6 * (32 * r_tmp ** 2 + 18 * r_tmp + 3) * (r_tmp < 1.0) # wenland set order 4
        wen_set_8[it] = (1 - r_tmp) ** 8 * (32 * r_tmp ** 3 + 25 * r_tmp ** 2 + 8 * r_tmp + 1) * (r_tmp < 1.0) # wenland set order 6
        wen_set_3[it] = (1 - r_tmp) ** 3 * (r_tmp < 1.0) # wenland set order 0
        wen_set_5[it] = (1 - r_tmp) ** 5 * (5 * r_tmp + 1) * (r_tmp < 1.0) # wenland set order 2
        wen_set_7[it] = (1 - r_tmp) ** 7 * (16 * r_tmp ** 2 + 7 * r_tmp + 1) * (r_tmp < 1.0) # wenland set order 4
    phi_set = np.nan_to_num(phi_set)
    truncated_psi_set = np.nan_to_num(truncated_psi_set)
    wen_set_2 = np.nan_to_num(wen_set_2)
    wen_set_4 = np.nan_to_num(wen_set_4)
    wen_set_6 = np.nan_to_num(wen_set_6)
    wen_set_8 = np.nan_to_num(wen_set_8)
    wen_set_3 = np.nan_to_num(wen_set_3)
    wen_set_5 = np.nan_to_num(wen_set_5)
    wen_set_7 = np.nan_to_num(wen_set_7)
    # Approximation
    A_gaus = np.zeros([n_bfs + 1, n_bfs + 1])
    A_moll = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_2 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_4 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_6 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_8 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_3 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_5 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_wen_7 = np.zeros([n_bfs + 1, n_bfs + 1])
    A_t_gaus = np.zeros([n_bfs + 1, n_bfs + 1])
    A_t_gaus_biases = np.zeros([2 * (n_bfs + 1), 2 * (n_bfs + 1)])
    b_gaus = np.zeros(n_bfs + 1)
    b_t_gaus_biases = np.zeros(2 * (n_bfs + 1))
    b_moll = np.zeros(n_bfs + 1)
    b_wen_2 = np.zeros(n_bfs + 1)
    b_wen_4 = np.zeros(n_bfs + 1)
    b_wen_6 = np.zeros(n_bfs + 1)
    b_wen_8 = np.zeros(n_bfs + 1)
    b_wen_3 = np.zeros(n_bfs + 1)
    b_wen_5 = np.zeros(n_bfs + 1)
    b_wen_7 = np.zeros(n_bfs + 1)
    b_t_gaus = np.zeros(n_bfs + 1)
    for h in range(n_bfs + 1):
        for k in range(h, n_bfs + 1):
            A_gaus[h][k] = integrate.simps(psi_set[h] * psi_set[k] / (np.sum(psi_set, 0) ** 2) * s_span ** 2, s_span)
            A_moll[h][k] = integrate.simps((phi_set[h] * phi_set[k] / (np.sum(phi_set, 0) ** 2) * s_span ** 2), s_span)
            A_wen_2[h][k] = integrate.simps((wen_set_2[h] * wen_set_2[k] / (np.sum(wen_set_2, 0) ** 2) * s_span ** 2), s_span)
            A_wen_4[h][k] = integrate.simps((wen_set_4[h] * wen_set_4[k] / (np.sum(wen_set_4, 0) ** 2) * s_span ** 2), s_span)
            A_wen_6[h][k] = integrate.simps((wen_set_6[h] * wen_set_6[k] / (np.sum(wen_set_6, 0) ** 2) * s_span ** 2), s_span)
            A_wen_8[h][k] = integrate.simps((wen_set_8[h] * wen_set_8[k] / (np.sum(wen_set_8, 0) ** 2) * s_span ** 2), s_span)
            A_wen_3[h][k] = integrate.simps((wen_set_3[h] * wen_set_3[k] / (np.sum(wen_set_3, 0) ** 2) * s_span ** 2), s_span)
            A_wen_5[h][k] = integrate.simps((wen_set_5[h] * wen_set_5[k] / (np.sum(wen_set_5, 0) ** 2) * s_span ** 2), s_span)
            A_wen_7[h][k] = integrate.simps((wen_set_7[h] * wen_set_7[k] / (np.sum(wen_set_7, 0) ** 2) * s_span ** 2), s_span)
            A_t_gaus[h][k] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span ** 2, s_span)
            A_gaus[k][h] = A_gaus[h][k]
            A_moll[k][h] = A_moll[h][k]
            A_wen_2[k][h] = A_wen_2[h][k]
            A_wen_4[k][h] = A_wen_4[h][k]
            A_wen_6[k][h] = A_wen_6[h][k]
            A_wen_8[k][h] = A_wen_8[h][k]
            A_wen_3[k][h] = A_wen_3[h][k]
            A_wen_5[k][h] = A_wen_5[h][k]
            A_wen_7[k][h] = A_wen_7[h][k]
            A_t_gaus_biases[k][h] = A_t_gaus_biases[h][k]
            A_t_gaus_biases[2 * h][2 * k] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span ** 2, s_span)
            A_t_gaus_biases[2 * h + 1][2 * k] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span, s_span)
            A_t_gaus_biases[2 * h][2 * k + 1] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span, s_span)
            A_t_gaus_biases[2 * h + 1][2 * k + 1] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2), s_span)
            A_t_gaus_biases[2 * k][2 * h] = A_t_gaus_biases[2 * h][2 * k]
            A_t_gaus_biases[2 * k][2 * h + 1] = A_t_gaus_biases[2 * h + 1][2 * k]
            A_t_gaus_biases[2 * k + 1][2 * h] = A_t_gaus_biases[2 * h][2 * k + 1]
            A_t_gaus_biases[2 * k + 1][2 * h + 1] = A_t_gaus_biases[2 * h][2 * k + 1]
        b_gaus[h] = integrate.simps(psi_set[h] * s_span * traj / np.sum(psi_set, 0), s_span)
        b_moll[h] = integrate.simps((phi_set[h] * s_span * traj / np.sum(phi_set, 0)), s_span)
        b_wen_2[h] = integrate.simps((wen_set_2[h] * s_span * traj / np.sum(wen_set_2, 0)), s_span)
        b_wen_4[h] = integrate.simps((wen_set_4[h] * s_span * traj / np.sum(wen_set_4, 0)), s_span)
        b_wen_6[h] = integrate.simps((wen_set_6[h] * s_span * traj / np.sum(wen_set_6, 0)), s_span)
        b_wen_8[h] = integrate.simps((wen_set_8[h] * s_span * traj / np.sum(wen_set_8, 0)), s_span)
        b_wen_3[h] = integrate.simps((wen_set_3[h] * s_span * traj / np.sum(wen_set_3, 0)), s_span)
        b_wen_5[h] = integrate.simps((wen_set_5[h] * s_span * traj / np.sum(wen_set_5, 0)), s_span)
        b_wen_7[h] = integrate.simps((wen_set_7[h] * s_span * traj / np.sum(wen_set_7, 0)), s_span)
        b_t_gaus[h] = integrate.simps(truncated_psi_set[h] * s_span * traj / np.sum(psi_set, 0), s_span)
        b_t_gaus_biases[2 * h] = integrate.simps(truncated_psi_set[h] * s_span * traj / np.sum(truncated_psi_set, 0), s_span)
        b_t_gaus_biases[2 * h + 1] = integrate.simps(truncated_psi_set[h] * traj / np.sum(truncated_psi_set, 0), s_span)
    # Compute the two set of weights:
    w_gaus = np.linalg.solve(A_gaus, b_gaus)
    w_moll = np.linalg.solve(A_moll, b_moll)
    w_wen_2 = np.linalg.solve(A_wen_2, b_wen_2)
    w_wen_4 = np.linalg.solve(A_wen_4, b_wen_4)
    w_wen_6 = np.linalg.solve(A_wen_6, b_wen_6)
    w_wen_8 = np.linalg.solve(A_wen_8, b_wen_8)
    w_wen_3 = np.linalg.solve(A_wen_3, b_wen_3)
    w_wen_5 = np.linalg.solve(A_wen_5, b_wen_5)
    w_wen_7 = np.linalg.solve(A_wen_7, b_wen_7)
    w_t_gaus = np.linalg.solve(A_t_gaus, b_t_gaus)
    eta_t_gaus_biased = np.linalg.solve(A_t_gaus_biases, b_t_gaus_biases)
    w_t_gaus_biases = eta_t_gaus_biased[0 : 2*(n_bfs + 1): 2]
    biases_t_gaus = eta_t_gaus_biased[1 : 2 * (n_bfs + 1) + 1 : 2]
    # Compute the approximants
    approx_gaus = np.dot(psi_set.transpose(), w_gaus) * s_span / np.sum(psi_set, 0)
    approx_moll = (np.dot(phi_set.transpose(), w_moll) * s_span / np.sum(phi_set, 0))
    approx_wen_2 = (np.dot(wen_set_2.transpose(), w_wen_2) * s_span / np.sum(wen_set_2, 0))
    approx_wen_4 = (np.dot(wen_set_4.transpose(), w_wen_4) * s_span / np.sum(wen_set_4, 0))
    approx_wen_6 = (np.dot(wen_set_6.transpose(), w_wen_6) * s_span / np.sum(wen_set_6, 0))
    approx_wen_8 = (np.dot(wen_set_8.transpose(), w_wen_8) * s_span / np.sum(wen_set_8, 0))
    approx_wen_3 = (np.dot(wen_set_3.transpose(), w_wen_3) * s_span / np.sum(wen_set_3, 0))
    approx_wen_5 = (np.dot(wen_set_5.transpose(), w_wen_5) * s_span / np.sum(wen_set_5, 0))
    approx_wen_7 = (np.dot(wen_set_7.transpose(), w_wen_7) * s_span / np.sum(wen_set_7, 0))
    approx_t_gaus = np.dot(truncated_psi_set.transpose(), w_t_gaus) * s_span / np.sum(truncated_psi_set, 0)
    approx_t_gaus_biases = (np.dot(truncated_psi_set.transpose(), w_t_gaus_biases) * s_span + np.dot(truncated_psi_set.transpose(), biases_t_gaus)) / np.sum(truncated_psi_set, 0)
    err_gaus[count] = np.sqrt(integrate.simps((approx_gaus - traj) ** 2, t_span))
    err_moll[count] = np.sqrt(integrate.simps((approx_moll - traj) ** 2, t_span))
    err_wen_2[count] = np.sqrt(integrate.simps((approx_wen_2 - traj) ** 2, t_span))
    err_wen_4[count] = np.sqrt(integrate.simps((approx_wen_4 - traj) ** 2, t_span))
    err_wen_6[count] = np.sqrt(integrate.simps((approx_wen_6 - traj) ** 2, t_span))
    err_wen_8[count] = np.sqrt(integrate.simps((approx_wen_8 - traj) ** 2, t_span))
    err_wen_3[count] = np.sqrt(integrate.simps((approx_wen_3 - traj) ** 2, t_span))
    err_wen_5[count] = np.sqrt(integrate.simps((approx_wen_5 - traj) ** 2, t_span))
    err_wen_7[count] = np.sqrt(integrate.simps((approx_wen_7 - traj) ** 2, t_span))
    err_t_g[count] = np.sqrt(integrate.simps((approx_t_gaus - traj) ** 2, t_span))
    err_t_gaus_biased[count] = np.sqrt(integrate.simps((approx_t_gaus_biases - traj) ** 2, t_span))

"""
Compute the error of biased truncated gaussian w.r.t. the number of parameters
"""
range_nbfs_param = 2 * (np.ceil(np.power(2, np.linspace(1,7,10))) / 2).astype(int)
err_t_gaus_n_param = np.zeros(len(range_nbfs_param))
count = -1
for n_bfs in (range_nbfs_param / 2).astype(int):
    count += 1
    truncated_psi_set = np.zeros([n_bfs + 1, 1000])
    c_i = np.exp(-alpha * np.arange(0, n_bfs + 1, 1) * T / n_bfs) # centers
    h_i = (np.diff(c_i)) ** (-2.0)
    h_i = np.append(h_i, h_i[-1]) # widths for gaussian
    for it in range(n_bfs+1):
        truncated_psi_set[it] = np.exp(- 0.5 * h_i[it] * (s_span - c_i[it]) ** 2) * ((s_span - c_i[it]) <= (K * h_i[it] ** (-0.5)))
    truncated_psi_set = np.nan_to_num(truncated_psi_set)
    A_t_gaus_biases = np.zeros([2 * (n_bfs + 1), 2 * (n_bfs + 1)])
    b_t_gaus_biases = np.zeros(2 * (n_bfs + 1))
    for h in range(n_bfs + 1):
        for k in range(h, n_bfs + 1):
            A_t_gaus_biases[k][h] = A_t_gaus_biases[h][k]
            A_t_gaus_biases[2 * h][2 * k] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span ** 2, s_span)
            A_t_gaus_biases[2 * h + 1][2 * k] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span, s_span)
            A_t_gaus_biases[2 * h][2 * k + 1] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2) * s_span, s_span)
            A_t_gaus_biases[2 * h + 1][2 * k + 1] = integrate.simps(truncated_psi_set[h] * truncated_psi_set[k] / (np.sum(truncated_psi_set, 0) ** 2), s_span)
            A_t_gaus_biases[2 * k][2 * h] = A_t_gaus_biases[2 * h][2 * k]
            A_t_gaus_biases[2 * k][2 * h + 1] = A_t_gaus_biases[2 * h + 1][2 * k]
            A_t_gaus_biases[2 * k + 1][2 * h] = A_t_gaus_biases[2 * h][2 * k + 1]
            A_t_gaus_biases[2 * k + 1][2 * h + 1] = A_t_gaus_biases[2 * h][2 * k + 1]
        b_t_gaus_biases[2 * h] = integrate.simps(truncated_psi_set[h] * s_span * traj / np.sum(truncated_psi_set, 0), s_span)
        b_t_gaus_biases[2 * h + 1] = integrate.simps(truncated_psi_set[h] * traj / np.sum(truncated_psi_set, 0), s_span)
    eta_t_gaus_biased = np.linalg.solve(A_t_gaus_biases, b_t_gaus_biases)
    w_t_gaus_biases = eta_t_gaus_biased[0 : 2*(n_bfs + 1): 2]
    biases_t_gaus = eta_t_gaus_biased[1 : 2 * (n_bfs + 1) + 1 : 2]
    approx_t_gaus_biases = (np.dot(truncated_psi_set.transpose(), w_t_gaus_biases) * s_span + np.dot(truncated_psi_set.transpose(), biases_t_gaus)) / np.sum(truncated_psi_set, 0)
    err_t_gaus_n_param[count] = np.sqrt(integrate.simps((approx_t_gaus_biases - traj) ** 2, t_span))

ms = 5 # markersize
lwidth = 1 # linewidth
fs = 14 # font size

fig2 = plt.figure(1)
ax = plt.subplot(111)
ax.loglog(range_nbfs, err_gaus, 'x--', lw = lwidth, markersize = ms, label = r'$\psi$')
ax.loglog(range_nbfs, err_t_g, '*--', lw = lwidth, markersize = ms, label = r'$\tilde{\psi}_U$')
ax.loglog(range_nbfs, err_t_gaus_biased, '+--', lw = lwidth, markersize = ms, label = r'$\tilde{\psi}_B$')
ax.loglog(range_nbfs, err_moll, 'o--', lw = lwidth, markersize = ms, label = r'$\varphi_i$')
ax.loglog(range_nbfs, err_wen_2, '<--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(2)}$')
ax.loglog(range_nbfs, err_wen_3, '1--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(3)}$')
ax.loglog(range_nbfs, err_wen_4, '>--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(4)}$')
ax.loglog(range_nbfs, err_wen_5, '2--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(5)}$')
ax.loglog(range_nbfs, err_wen_6, 'v--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(6)}$')
ax.loglog(range_nbfs, err_wen_7, '3--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(7)}$')
ax.loglog(range_nbfs, err_wen_8, '^--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(8)}$')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fontsize = fs)
plt.xlabel(r'Number of basis functions')

fig4 = plt.figure(2)
ax = plt.subplot(111)
ax.loglog(range_nbfs, err_gaus, 'x--', lw = lwidth, markersize = ms, label = r'$\psi$')
ax.loglog(range_nbfs, err_t_g, '*--', lw = lwidth, markersize = ms, label = r'$\tilde{\psi}_U$')
ax.loglog(range_nbfs_param, err_t_gaus_n_param, '+--', lw = lwidth, markersize = ms, label = r'$\tilde{\psi}_B$')
ax.loglog(range_nbfs, err_moll, 'o--', lw = lwidth, markersize = ms, label = r'$\varphi_i$')
ax.loglog(range_nbfs, err_wen_2, '<--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(2)}$')
ax.loglog(range_nbfs, err_wen_3, '1--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(3)}$')
ax.loglog(range_nbfs, err_wen_4, '>--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(4)}$')
ax.loglog(range_nbfs, err_wen_5, '2--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(5)}$')
ax.loglog(range_nbfs, err_wen_6, 'v--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(6)}$')
ax.loglog(range_nbfs, err_wen_7, '3--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(7)}$')
ax.loglog(range_nbfs, err_wen_8, '^--', lw = lwidth, markersize = ms, label = r'$\phi_i^{(8)}$')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fontsize = fs)
plt.xlabel(r'number of parameters')
plt.show()