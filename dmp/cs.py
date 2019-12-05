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

class CanonicalSystem():
    '''
    Implementation of the canonical dynamical system
    '''

    def __init__(self, dt, alpha_s = 4.0, run_time = 1.0, **kwargs):

        # Parameter setting
        self.alpha_s = alpha_s
        self.run_time = run_time
        self.dt = dt
        self.timesteps = int(self.run_time / self.dt) + 1
        self.reset_state()

    def reset_state(self):
        '''
        Reset the system state
        '''

        self.s = 1.0

    def rollout(self, tau = 1.0, **kwargs):
        '''
        Generate s.
        '''

        timesteps = int(self.timesteps / tau)
        s_track = np.zeros(timesteps)
        self.reset_state()
        for t in range(timesteps):
            s_track[t] = self.s
            self.step(tau = tau)
        return s_track

    def step(self, tau = 1.0, error_coupling = 1.0, **kwargs):
        '''
        Generate a single step of s.
          tau float
          error_coupling float
        '''

        # Since the canonical system is linear, we use an exponential method
        # (which is exact)
        const = - self.alpha_s / tau / error_coupling
        self.s *= np.exp(const * self.dt)
        return self.s