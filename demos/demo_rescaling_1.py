import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from dmp import dmp_cartesian as dmp

# Creation of the trajectory
t_steps = 10 ** 3
t_des = np.linspace (0., 2 * np.pi, t_steps)
x_1 = np.cos(t_des) * t_des ** 2.0
x_2 = np.sin(t_des) * t_des
x_des = np.zeros([2, t_steps])
x_des[0,:] = x_1
x_des[1,:] = x_2
original_goal = x_des[:, -1]
print_legend = False
# Plot stuff
lw = 1 # line width
ms = 10 # marker size

# --------------- #
#  Case K = 1000  #
# --------------- #

# Initialization and lLearning the forcing term
myK = 1000.0
alpha_s = 4.0
dmp_rescaling = dmp.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = 'rotodilatation', alpha_s = alpha_s, tol = 1)
dmp_classical = dmp.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = None, alpha_s = alpha_s, tol = 1)
dmp_rescaling.imitate_path (x_des = x_des.transpose(), t_des = t_des)
dmp_classical.imitate_path (x_des = x_des.transpose(), t_des = t_des)
x_learned, _, _, t_learned = dmp_classical.rollout()

# Only rotation of the reference frame
theta = 135 * np.pi / 180.0
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
new_goal_rot = np.dot(R, original_goal)
dmp_classical.x_goal = new_goal_rot.copy()
dmp_rescaling.x_goal = new_goal_rot.copy()
x_rot_classical, _, _, t_rot_classical = dmp_classical.rollout()
x_rot_rescaling, _, _, t_rot_rescaling = dmp_rescaling.rollout()

# Only dilatation of the reference frame
new_goal_dilat = original_goal * 2.0
dmp_classical.x_goal = new_goal_dilat.copy()
dmp_rescaling.x_goal = new_goal_dilat.copy()
x_dilat_classical, _, _, t_dilat_classical = dmp_classical.rollout()
x_dilat_rescaling, _, _, t_dilat_rescaling = dmp_rescaling.rollout()

# Only shrinking of the reference frame
new_goal_shrink = original_goal / 2.5
dmp_classical.x_goal = new_goal_shrink.copy()
dmp_rescaling.x_goal = new_goal_shrink.copy()
x_shrink_classical, _, _, t_shrink_classical = dmp_classical.rollout()
x_shrink_rescaling, _, _, t_shrink_rescaling = dmp_rescaling.rollout()

# Demo with changing of goal position
new_goal_move = original_goal
dmp_classical.x_goal = new_goal_move.copy()
dmp_rescaling.x_goal = new_goal_move.copy()
goal_track = np.zeros([0, 2])
goal_track = np.append(goal_track, [original_goal], axis = 0)
# Goal evolution
theta = np.pi / 2 / dmp_classical.cs.timesteps
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
R *= 1 + 0.5 / dmp_classical.cs.timesteps
# Rollout step by step
dmp_classical.reset_state()
dmp_rescaling.reset_state()
# Set up tracking vectors
x_track_classical = np.zeros((0, dmp_classical.n_dmps))
dx_track_classical = np.zeros((0, dmp_classical.n_dmps))
ddx_track_classical = np.zeros((0, dmp_classical.n_dmps))
x_track_rescaling = np.zeros((0, dmp_rescaling.n_dmps))
dx_track_rescaling = np.zeros((0, dmp_rescaling.n_dmps))
ddx_track_rescaling = np.zeros((0, dmp_rescaling.n_dmps))
x_track_classical = np.append(x_track_classical, [dmp_classical.x_0], axis = 0)
dx_track_classical = np.append(dx_track_classical, [0.0 * dmp_classical.x_0], axis = 0)
dx_track_classical = np.append(ddx_track_classical, [0.0 * dmp_classical.x_0], axis = 0)
x_track_rescaling = np.append(x_track_rescaling, [dmp_rescaling.x_0], axis = 0)
dx_track_rescaling = np.append(dx_track_rescaling, [0.0 * dmp_rescaling.x_0], axis = 0)
dx_track_rescaling = np.append(ddx_track_rescaling, [0.0 * dmp_rescaling.x_0], axis = 0)
# Rollout of classical DMP
flag = False
t = 0
while (not flag):
  # Run and record timestep
  x_track_s, dx_track_s, ddx_track_s = dmp_classical.step(tau = 1)
  x_track_classical = np.append(x_track_classical, [x_track_s], axis=0)
  dx_track_classical = np.append(dx_track_classical, [dx_track_s],axis=0)
  ddx_track_classical = np.append(ddx_track_classical, [ddx_track_s],axis=0)
  if (t < dmp_classical.cs.timesteps / 2):
    dmp_classical.x_goal = np.dot(R, dmp_classical.x_goal)
  goal_track = np.append(goal_track, [dmp_classical.x_goal], axis = 0)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_classical.x_goal)
  err_rel = err_abs / (np.linalg.norm(dmp_classical.x_goal - dmp_classical.x_0) + 1e-14)
  flag = ((t >= dmp_classical.cs.timesteps) and err_rel <= dmp_classical.tol)
t_track_classical = np.linspace(0, dmp_classical.cs.dt * t, t+1)
# Rollout of DMP++
flag = False
t = 0
while (not flag):
  # Run and record timestep
  x_track_s, dx_track_s, ddx_track_s = dmp_rescaling.step(tau = 1)
  x_track_rescaling = np.append(x_track_rescaling, [x_track_s], axis=0)
  dx_track_rescaling = np.append(dx_track_rescaling, [dx_track_s],axis=0)
  ddx_track_rescaling = np.append(ddx_track_rescaling, [ddx_track_s],axis=0)
  if (t < dmp_classical.cs.timesteps / 2):
    dmp_rescaling.x_goal = np.dot(R, dmp_rescaling.x_goal)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_rescaling.x_goal)
  err_rel = err_abs / (np.linalg.norm(dmp_rescaling.x_goal - dmp_rescaling.x_0) + 1e-14)
  flag = ((t >= dmp_rescaling.cs.timesteps) and err_rel <= dmp_rescaling.tol)
t_track_rescaling = np.linspace(0, dmp_rescaling.cs.dt * t, t+1)

# -------------- #
#  Case K = 100  #
# -------------- #

# Initialization and lLearning the forcing term
myK = 100.0
alpha_s = 4.0
dmp_rescaling = dmp.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = 'rotodilatation', alpha_s = alpha_s, tol = 1)
dmp_classical = dmp.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = None, alpha_s = alpha_s, tol = 1)
dmp_rescaling.imitate_path (x_des = x_des.transpose(), t_des = t_des)
dmp_classical.imitate_path (x_des = x_des.transpose(), t_des = t_des)
x_learned, _, _,_ = dmp_classical.rollout()

# Only rotation of the reference frame
theta = 135 * np.pi / 180.0
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
new_goal = np.dot(R, original_goal)
dmp_classical.x_goal = new_goal.copy()
dmp_rescaling.x_goal = new_goal.copy()
x_rot_classical_newK, _, _, t_rot_classical_newK = dmp_classical.rollout()
x_rot_rescaling_newK, _, _, t_rot_rescaling_newK = dmp_rescaling.rollout()

# Only dilatation of the reference frame
new_goal = original_goal * 2.0
dmp_classical.x_goal = new_goal.copy()
dmp_rescaling.x_goal = new_goal.copy()
x_dilat_classical_newK, _, _, t_dilat_classical_newK = dmp_classical.rollout()
x_dilat_rescaling_newK, _, _, t_dilat_rescaling_newK = dmp_rescaling.rollout()

# Only shrinking of the reference frame
new_goal = original_goal / 2.5
dmp_classical.x_goal = new_goal.copy()
dmp_rescaling.x_goal = new_goal.copy()
x_shrink_classical_newK, _, _, t_shrink_classical_newK = dmp_classical.rollout()
x_shrink_rescaling_newK, _, _, t_shrink_rescaling_newK = dmp_rescaling.rollout()

# Demo with changing of goal position
new_goal = original_goal
dmp_classical.x_goal = new_goal.copy()
dmp_rescaling.x_goal = new_goal.copy()
goal_track = np.zeros([0, 2])
goal_track = np.append(goal_track, [original_goal], axis = 0)
# Goal evolution
theta = np.pi / 2 / dmp_classical.cs.timesteps
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
R *= 1 + 0.5 / dmp_classical.cs.timesteps
# Rollout step by step
dmp_classical.reset_state()
dmp_rescaling.reset_state()
# Set up tracking vectors
x_track_classical_newK = np.zeros((0, dmp_classical.n_dmps))
dx_track_classical = np.zeros((0, dmp_classical.n_dmps))
ddx_track_classical = np.zeros((0, dmp_classical.n_dmps))
x_track_rescaling_newK = np.zeros((0, dmp_rescaling.n_dmps))
dx_track_rescaling = np.zeros((0, dmp_rescaling.n_dmps))
ddx_track_rescaling = np.zeros((0, dmp_rescaling.n_dmps))
x_track_classical_newK = np.append(x_track_classical_newK, [dmp_classical.x_0], axis = 0)
dx_track_classical = np.append(dx_track_classical, [0.0 * dmp_classical.x_0], axis = 0)
dx_track_classical = np.append(ddx_track_classical, [0.0 * dmp_classical.x_0], axis = 0)
x_track_rescaling_newK = np.append(x_track_rescaling_newK, [dmp_rescaling.x_0], axis = 0)
dx_track_rescaling = np.append(dx_track_rescaling, [0.0 * dmp_rescaling.x_0], axis = 0)
dx_track_rescaling = np.append(ddx_track_rescaling, [0.0 * dmp_rescaling.x_0], axis = 0)
# Rollout of classical DMP
flag = False
t = 0
while (not flag):
  # Run and record timestep
  x_track_s, dx_track_s, ddx_track_s = dmp_classical.step(tau = 1)
  x_track_classical_newK = np.append(x_track_classical_newK, [x_track_s], axis=0)
  dx_track_classical = np.append(dx_track_classical, [dx_track_s],axis=0)
  ddx_track_classical = np.append(ddx_track_classical, [ddx_track_s],axis=0)
  if (t < dmp_classical.cs.timesteps / 2):
    dmp_classical.x_goal = np.dot(R, dmp_classical.x_goal)
  goal_track = np.append(goal_track, [dmp_classical.x_goal], axis = 0)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_classical.x_goal)
  err_rel = err_abs / (np.linalg.norm(dmp_classical.x_goal - dmp_classical.x_0) + 1e-14)
  flag = ((t >= dmp_classical.cs.timesteps) and err_rel <= dmp_classical.tol)
t_track_classical = np.linspace(0, dmp_classical.cs.dt * t, t+1)
# Rollout of DMP++
flag = False
t = 0
while (not flag):
  # Run and record timestep
  x_track_s, dx_track_s, ddx_track_s = dmp_rescaling.step(tau = 1)
  x_track_rescaling_newK = np.append(x_track_rescaling_newK, [x_track_s], axis=0)
  dx_track_rescaling = np.append(dx_track_rescaling, [dx_track_s],axis=0)
  ddx_track_rescaling = np.append(ddx_track_rescaling, [ddx_track_s],axis=0)
  if (t < dmp_classical.cs.timesteps / 2):
    dmp_rescaling.x_goal = np.dot(R, dmp_rescaling.x_goal)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_rescaling.x_goal)
  err_rel = err_abs / (np.linalg.norm(dmp_rescaling.x_goal - dmp_rescaling.x_0) + 1e-14)
  flag = ((t >= dmp_rescaling.cs.timesteps) and err_rel <= dmp_rescaling.tol)
t_track_rescaling = np.linspace(0, dmp_rescaling.cs.dt * t, t+1)

# ---------- #
#  Plotting  #
# ---------- #

plt.figure(1)
plt.plot(x_learned[:, 0], x_learned[:, 1], '--b')
plt.plot(x_rot_classical[:, 0], x_rot_classical[:, 1], '-.g')
plt.plot(x_rot_classical_newK[:, 0], x_rot_classical_newK[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2)), color = 'purple')
plt.plot(x_rot_rescaling[:, 0], x_rot_rescaling[:, 1], '-r')
plt.plot(x_rot_rescaling_newK[:, 0], x_rot_rescaling_newK[:, 1], linestyle='dotted', color='orange')
plt.plot(new_goal_rot[0], new_goal_rot[1], '*k', markersize = ms, label = r'New goal position')
plt.plot(0, 0, '.k', markersize=ms)
plt.xlabel(r'$ x_1 $', fontsize=14)
plt.ylabel(r'$ x_2 $', fontsize=14)
plt.axis('equal')

plt.figure(2)
plt.plot(x_learned[:, 0], x_learned[:, 1], '--b')
plt.plot(x_dilat_classical[:, 0], x_dilat_classical[:, 1], '-.g')
plt.plot(x_dilat_classical_newK[:, 0], x_dilat_classical_newK[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2)), color = 'purple')
plt.plot(x_dilat_rescaling[:, 0], x_dilat_rescaling[:, 1], '-r')
plt.plot(x_dilat_rescaling_newK[:, 0], x_dilat_rescaling_newK[:, 1], linestyle='dotted', color='orange')
plt.plot(new_goal_dilat[0], new_goal_dilat[1], '*k', markersize = ms, label = r'New goal position')
plt.plot(0, 0, '.k', markersize=ms)
plt.xlabel(r'$ x_1 $', fontsize=14)
plt.ylabel(r'$ x_2 $', fontsize=14)
plt.axis('equal')

plt.figure(3)
plt.plot(x_learned[:, 0], x_learned[:, 1], '--b')
plt.plot(x_shrink_classical[:, 0], x_shrink_classical[:, 1], '-.g')
plt.plot(x_shrink_classical_newK[:, 0], x_shrink_classical_newK[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2)), color = 'purple')
plt.plot(x_shrink_rescaling[:, 0], x_shrink_rescaling[:, 1], '-r')
plt.plot(x_shrink_rescaling_newK[:, 0], x_shrink_rescaling_newK[:, 1], linestyle='dotted', color='orange')
plt.plot(new_goal_shrink[0], new_goal_shrink[1], '*k', markersize = ms, label = r'New goal position')
plt.plot(0, 0, '.k', markersize=ms)
plt.xlabel(r'$ x_1 $', fontsize=14)
plt.ylabel(r'$ x_2 $', fontsize=14)
plt.axis('equal')

plt.figure(4)
plt.plot(x_learned[:, 0], x_learned[:, 1], '--b')
plt.plot(x_track_classical[:, 0], x_track_classical[:, 1], '-.g')
plt.plot(x_track_classical_newK[:, 0], x_track_classical_newK[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2, 1, 2)), color = 'purple')
plt.plot(x_track_rescaling[:, 0], x_track_rescaling[:, 1], '-r')
plt.plot(x_track_rescaling_newK[:, 0], x_track_rescaling_newK[:, 1], linestyle='dotted', color='orange')
goal_range = list((np.linspace(0, len(goal_track[:, 0]) / 2, 15)).astype(int))
goal_range = goal_range[:-1]
plt.plot(goal_track[goal_range, 0], goal_track[goal_range, 1], '*k', markersize=(ms // 2))
plt.plot(goal_track[-1][0], goal_track[-1][1], '*k', markersize = ms, label = r'New goal position')
plt.plot(0, 0, '.k', markersize=ms)
plt.xlabel(r'$ x_1 $', fontsize=14)
plt.ylabel(r'$ x_2 $', fontsize=14)
plt.axis('equal')

plt.show()