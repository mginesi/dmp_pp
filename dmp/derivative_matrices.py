'''
Copyright (C) 2019 Michele Ginesi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import scipy.sparse as sparse

def compute_D1(n, dt):
    '''
    Compute the matrices used to estimate the first derivative.
      n float  : dimensionality
      dt float : timestep
    '''

    d1_p = np.ones([n - 1])
    d1_p[0] = 4.
    d1_m = - np.ones([n-1])
    d1_m[-1] = - 4.
    D1 = sparse.diags(np.array([d1_p, d1_m]), [1, -1]).toarray()
    D1[0,0] = - 3.
    D1[0, 2] = -1.
    D1[-1, -3] = 1.
    D1[-1,-1] = 3.
    D1 /= 2 * dt
    return D1

def compute_D2(n, dt):
    '''
    Compute the matrices used to estimate the first derivative.
      n float  : dimensionality
      dt float : timestep
    '''

    d2_p = np.ones([n-1])
    d2_p[0] = -5.
    d2_m = np.ones([n-1])
    d2_m[-1] = -5.
    d2_c = - 2. * np.ones([n])
    d2_c[0] = 2.
    d2_c[-1] = 2.
    D2 = sparse.diags(np.array([d2_p, d2_c, d2_m]), [1, 0, -1]).toarray()
    D2[0, 2] = 4.
    D2[0, 3] = -1.
    D2[-1, -3] = 4.
    D2[-1, -4] = -1.
    D2 /= dt ** 2
    return D2