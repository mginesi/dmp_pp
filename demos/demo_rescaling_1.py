import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# To use the package in the main folder
import sys
sys.path.insert(0, '../codes')
sys.path.insert(0, 'codes/')

import dmp_cartesian

# Debugger
import pdb

# Creation of the trajectory
t_steps = 10 ** 3
t = np.linspace (0.0, 2.0 * np.pi, t_steps)
x_1 = np.cos(t) * t * t
x_2 = np.sin(t) * t
x_des = np.zeros([t_steps, 2])
x_des[:, 0] = x_1
x_des[:, 1] = x_2
original_goal = x_des[-1]
print_legend = False
# Plot stuff
lw = 1 # line width
ms = 10 # marker size

## Case K = 1000

# Initialization and lLearning the forcing term
myK = 1000
alpha_s = 4.0
dmp_rescaling = dmp_cartesian.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = True, alpha_s = alpha_s, tol = 0.05)
dmp_classical = dmp_cartesian.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = False, alpha_s = alpha_s, tol = 0.05)
dmp_rescaling.imitate_path (x_des = x_des)
dmp_classical.imitate_path (x_des = x_des)
x_learned, _, _, _ = dmp_classical.rollout()

# Only rotation of the reference frame
theta = 135 * np.pi / 180.0
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
new_goal = np.dot(original_goal, R.transpose())
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_rot_classical, _, _, _ = dmp_classical.rollout()
x_rot_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(1)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_rot_classical[:, 0], x_rot_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_rot_rescaling[:, 0], x_rot_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 4')

# Only dilatation of the reference frame
new_goal = original_goal * 1.5
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_dilat_classical, _, _, _ = dmp_classical.rollout()
x_dilat_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(2)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_dilat_classical[:, 0], x_dilat_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_dilat_rescaling[:, 0], x_dilat_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 4')

# Only shrinking of the reference frame
new_goal = original_goal / 2.5
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_shrink_classical, _, _, _ = dmp_classical.rollout()
x_shrink_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(3)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_shrink_classical[:, 0], x_shrink_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_shrink_rescaling[:, 0], x_shrink_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 4')

# Demo with changing of goal position
new_goal = original_goal
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
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
x_track_classical = np.append(x_track_classical, [dmp_classical.x0], axis = 0)
dx_track_classical = np.append(dx_track_classical, [0.0 * dmp_classical.x0], axis = 0)
dx_track_classical = np.append(ddx_track_classical, [0.0 * dmp_classical.x0], axis = 0)
x_track_rescaling = np.append(x_track_rescaling, [dmp_rescaling.x0], axis = 0)
dx_track_rescaling = np.append(dx_track_rescaling, [0.0 * dmp_rescaling.x0], axis = 0)
dx_track_rescaling = np.append(ddx_track_rescaling, [0.0 * dmp_rescaling.x0], axis = 0)
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
    dmp_classical.goal = np.dot(R, dmp_classical.goal)
  goal_track = np.append(goal_track, [dmp_classical.goal], axis = 0)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_classical.goal)
  err_rel = err_abs / (np.linalg.norm(dmp_classical.goal - dmp_classical.x0) + 1e-14)
  flag = ((t >= dmp_classical.cs.timesteps) and err_rel <= dmp_classical.tol)
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
    dmp_rescaling.goal = np.dot(R, dmp_rescaling.goal)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_rescaling.goal)
  err_rel = err_abs / (np.linalg.norm(dmp_rescaling.goal - dmp_rescaling.x0) + 1e-14)
  flag = ((t >= dmp_rescaling.cs.timesteps) and err_rel <= dmp_rescaling.tol)

plt.figure(4)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_track_classical[:, 0], x_track_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_track_rescaling[:, 0], x_track_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(goal_track[:, 0], goal_track[:,1], ':k', linewidth = lw, label = r'New goal position')
plt.plot(goal_track[-1][0], goal_track[-1][1], '*k', markersize = ms)
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 4')

## Case K = 100

# Initialization and learning the forcing term
myK = 100
alpha_s = 4.0
dmp_rescaling = dmp_cartesian.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = True, alpha_s = alpha_s, tol = 0.05)
dmp_classical = dmp_cartesian.DMPs_cartesian (n_dmps = 2, n_bfs = 50, K = myK, rescale = False, alpha_s = alpha_s, tol = 0.05)
dmp_rescaling.imitate_path (x_des = x_des)
dmp_classical.imitate_path (x_des = x_des)
x_learned, _, _, _ = dmp_classical.rollout()

# Only rotation of the reference frame
theta = 135 * np.pi / 180.0
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
new_goal = np.dot(original_goal, R.transpose())
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_rot_classical, _, _, _ = dmp_classical.rollout()
x_rot_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(5)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_rot_classical[:, 0], x_rot_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_rot_rescaling[:, 0], x_rot_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 1')

# Only dilatation of the reference frame
new_goal = original_goal * 1.5
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_dilat_classical, _, _, _ = dmp_classical.rollout()
x_dilat_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(6)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_dilat_classical[:, 0], x_dilat_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_dilat_rescaling[:, 0], x_dilat_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 1')

# Only shrinking of the reference frame
new_goal = original_goal / 2.5
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
x_shrink_classical, _, _, _ = dmp_classical.rollout()
x_shrink_rescaling, _, _, _ = dmp_rescaling.rollout()
plt.figure(7)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_shrink_classical[:, 0], x_shrink_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_shrink_rescaling[:, 0], x_shrink_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(new_goal[0], new_goal[1], '*k', markersize = ms, label = r'New goal position')
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 1')

# Demo with changing of goal position
new_goal = original_goal
dmp_classical.goal = new_goal.copy()
dmp_rescaling.goal = new_goal.copy()
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
x_track_classical = np.append(x_track_classical, [dmp_classical.x0], axis = 0)
dx_track_classical = np.append(dx_track_classical, [0.0 * dmp_classical.x0], axis = 0)
dx_track_classical = np.append(ddx_track_classical, [0.0 * dmp_classical.x0], axis = 0)
x_track_rescaling = np.append(x_track_rescaling, [dmp_rescaling.x0], axis = 0)
dx_track_rescaling = np.append(dx_track_rescaling, [0.0 * dmp_rescaling.x0], axis = 0)
dx_track_rescaling = np.append(ddx_track_rescaling, [0.0 * dmp_rescaling.x0], axis = 0)
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
    dmp_classical.goal = np.dot(R, dmp_classical.goal)
  goal_track = np.append(goal_track, [dmp_classical.goal], axis = 0)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_classical.goal)
  err_rel = err_abs / (np.linalg.norm(dmp_classical.goal - dmp_classical.x0) + 1e-14)
  flag = ((t >= dmp_classical.cs.timesteps) and err_rel <= dmp_classical.tol)
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
    dmp_rescaling.goal = np.dot(R, dmp_rescaling.goal)
  t += 1
  err_abs = np.linalg.norm(x_track_s - dmp_rescaling.goal)
  err_rel = err_abs / (np.linalg.norm(dmp_rescaling.goal - dmp_rescaling.x0) + 1e-14)
  flag = ((t >= dmp_rescaling.cs.timesteps) and err_rel <= dmp_rescaling.tol)

plt.figure(8)
plt.plot(x_learned[:, 0], x_learned[:, 1], '-b', linewidth = lw, label = r'Demonstration')
plt.plot(x_track_classical[:, 0], x_track_classical[:, 1], '--r', linewidth = lw, label = r'Classical DMP')
plt.plot(x_track_rescaling[:, 0], x_track_rescaling[:, 1], '-.g', linewidth = lw, label = r'DMP++')
plt.plot(goal_track[:, 0], goal_track[:,1], ':k', linewidth = lw, label = r'New goal position')
plt.plot(goal_track[-1][0], goal_track[-1][1], '*k', markersize = ms)
if print_legend:
  plt.legend(loc = 'upper right')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')
plt.legend(loc = 'best')
plt.title(r'$\alpha$ = 1')

plt.show()