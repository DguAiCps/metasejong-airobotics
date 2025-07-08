# === mpc_controller.py ===

import numpy as np
import scipy.optimize

class MPCController:
    def __init__(self, dt=0.2, N=7, v_min=-2.0, v_max=2.0, w_min=-1.0, w_max=1.0, min_obs_dist=0.7):
        self.dt = dt
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.min_obs_dist = min_obs_dist

    def _cost(self, u_flat, x0, goal):
        u = u_flat.reshape(self.N, 2)
        x = np.array(x0)
        total = 0
        for k in range(self.N):
            v, w = u[k]
            x = x + self.dt * np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])
            total += 10 * np.sum((x[:2] - goal[:2])**2) + 0.1 * (v**2 + w**2)
            if k > 0:
                total += 0.5 * ((u[k][0] - u[k-1][0])**2 + (u[k][1] - u[k-1][1])**2)
        total += 10 * ((x[2] - goal[2])**2)
        return total

    def _constraint_factory(self, obs, k, x0):
        def constr(u_flat):
            u = u_flat.reshape(self.N, 2)
            x = np.array(x0)
            for i in range(k+1):
                v, w = u[i]
                x = x + self.dt * np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])
            return np.linalg.norm(x[:2] - obs) - self.min_obs_dist
        return constr

    def solve(self, x0, goal, obstacles):
        u0 = np.zeros((self.N, 2))
        bounds = [(self.v_min, self.v_max), (self.w_min, self.w_max)] * self.N
        cons = []

        if obstacles:
            obstacles = sorted(obstacles, key=lambda obs: np.sum((np.array(x0[:2]) - np.array(obs))**2))[:10]
            for k in range(self.N):
                for obs in obstacles:
                    cons.append({'type': 'ineq', 'fun': self._constraint_factory(obs, k, x0)})

        res = scipy.optimize.minimize(
            self._cost, u0.flatten(), args=(x0, goal),
            bounds=bounds, constraints=cons, method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-2}
        )

        u_opt = res.x[:2]
        return float(np.clip(u_opt[0], self.v_min, self.v_max)), float(np.clip(u_opt[1], self.w_min, self.w_max))
