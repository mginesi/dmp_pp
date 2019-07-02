# README #

This repository contains the implementation of Dynamic Movement Primitive Plus Plus (DMP++), in Python 3.5.

In particular, this repository contains all the synthetic tests done for the work (currently under revision):

_Ginesi M., Sansonetto N., and Fiorini P._; **DMP++: Overcoming Some Drawbacks of Dynamic Movement Primitives**

## Contents ##

This repository contains two folders, namely _codes/_ and _demos/_.
The _codes/_ folder contains all the functions needed to implement DMP++, while the _demo/_ folder contains the scripts used to perform the tests presented on the paper.

#### _codes/_ ####

_codes/_ contain the following files:
* _cs.py_ implements the Canonical System class, together with its methods.
* _derivative_matrices.py_ implements the following two functions:
  * `compute_D1(n, dt)` returns the matrix which discretize the first derivative of a 1D function discretized on an equispaced time domain of `n` points and `dt` timestep, using a second orde estimate;
  * `compute_D2(n, dt)` returns the matrix which discretize the second derivative of a 1D function discretized on an equispaced time domain of `n` points and `dt` timestep, using a second orde estimate.
* _dmp_cartesian.py_ implements the DMP++ class, together with its methods.
* _exponential_integrator.py_ implements the functions needed to perform an integration step using the Exponential Euler method. In particular the function `exp_eul_step(y, A, b, dt)` returns the solution at time $ n+1$, computed as $ y_{n+1} = y_n + k \varphi_1(k A) (A y_n + b(t_n)) $ for the problem $ \dot{y} = A y + b(t) $, with $y_n$ = `y`, $A$ = `A`, $b(t_n)$ = `b`, and $k$ = `dt`.
* _rotation_matri.py_ implements the functions needed to compute the roto-dilatation matrix. In particular, `roto_dilatation(x0, x1)` returns the roto-dilatation matrix which maps `x0` to `x1`.

#### _demos/_ ####

_demos/_ contain the following files:
* _demo_basis.py_ tests the accuracy in the approximation of a given function using different types of basis functions (Gaussian, trucated Gaussians, Wendland, and Mollifier-like). See Figure 2 of the paper.
* _demo_regression.py_ tests the learning-from-multible-observations porcess using a set of trajecotries obtained by integrating a known dynamical system. See Figure 4a-4b of the paper.
* _demo_rescaling.py_ tests the robustness of DMP++ against modification of the relative position between starting and ending points. See Figure 3 of the paper.

## Contact ##

To contact me please use one of the following mail addresses:

* michele.ginesi@gmail.com
* michele.ginesi@univr.it