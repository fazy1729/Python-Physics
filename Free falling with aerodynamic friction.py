import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


m = 2
g = -9.81
c = 0.5

def ecuatie (t, v):
    dv_dt = g - c*v**2/m
    return dv_dt

v0 = 0
t_span = (0,10)
y0 =[v0] 

sol = solve_ivp(ecuatie, t_span , y0, method='RK45', t_eval = np.linspace(0,10, 1000))

plt.plot(sol.t, sol.y[0], label = 'Ecuatia caderii corpului')
plt.yticks(np.arange(-2000,500, 500))
plt.xlabel('t')
plt.ylabel('v')
plt.grid()
plt.legend()
plt.show()
