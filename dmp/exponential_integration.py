# ---------------------------------------------------------------------------- #
# Copyright (C) 2018 Michele Ginesi

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #

import numpy as np

def log2(x):
    # ------------------------------------------------------------------------ #
    # Return a couple of values (f, e) such that
    #   x = f * 2^e
    # with 0.5 <= f < 1.
    # ------------------------------------------------------------------------ #
    sign = 1.0
    f = x
    e = int(0)
    if (x == 0):
        f = 0.0
        e = int(0)
    else:
        if (x < 0):
            x *= - 1.0
            f *= - 1.0
            sign = - 1.0
        if (x >= 1.0):
            e = int(0)
            while (f >= 1.0):
                f /= 2.0
                e += int(1)
        elif (x < 0.5):
            e = int(0)
            while (f < 0.5):
                f *= 2.0
                e -= int(1)
        f *= sign
    return f, e

def phi1(A):
    # ------------------------------------------------------------------------ #
    # Compute the phi1 function of a matrix A. The function phi_1 is defined as
    #               exp(z) - 1    +oo    z^j
    #   phi_1(z) = ------------ = sum ---------
    #                    z        j=0  (j + 1)!
    # We will use a Pade' approximation
    # ------------------------------------------------------------------------ #

    # Scale A by power of two so that its norm is < 0.5
    _, e = log2(np.linalg.norm(A, 1))
    s = np.min([np.max([0, e + 1]), 1023])

    # A power of 2 is representable exactly in binary arithmetic, thus we do not
    # introduce round errors
    A = A / (2.0 ** s)
    ID = np.eye(np.shape(A)[0])

    # Pade' coefficients
    n = np.array([1.0, 1.0 / 30, 1.0 / 30, 1.0 / 936, 1.0 / 4680, 1.0 / 171600,
        1.0 / 3603600, 1.0 / 259459200])
    d = np.array([1.0, - 7.0 / 15,  1.0 / 10, - 1.0 / 78, 1.0 / 936, -1/17160,
        1/514800, - 1.0 / 32432400])
    q = np.size(n)
    N = n[0] * ID
    D = d[0] * ID
    X = ID.copy()
    for ii in range(1, q):
        X = np.dot(A, X)
        N = N + n[ii] * X
        D = D + d[ii] * X
    N = np.dot(np.linalg.inv(D), N)

    # Undo scaling by repeating squaring
    phi0 = np.dot(A, N) + ID # will be the exponent
    for ii in range(s):
        N = np.dot((phi0 + ID), N / 2.0)
        phi0 = np.dot(phi0, phi0)
    return N

def exp_eul_step(y, A, b, dt):
    # ------------------------------------------------------------------------ #
    # Make a step of the exponential Euler method
    #   y_{n+1} = y_n + dt * phi1(dt * A) (A * y_n + b(t_n))
    # for a problem
    #   y' = A y + b(t)
    # ------------------------------------------------------------------------ #
    A_tilde = phi1(dt * A)
    b_tilde = np.dot(A, y) + b
    return y + dt * np.dot(A_tilde, b_tilde)