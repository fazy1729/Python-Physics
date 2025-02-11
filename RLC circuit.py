import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Definirea simbolurilor pentru constante
L = sp.Symbol('L', real=True, positive=True)
R = sp.Symbol('R', real=True, positive=True)
C = sp.Symbol('C', real=True, positive=True)
V_0 = sp.Symbol('V_0', real=True, positive=True)
t = sp.Symbol('t', real=True)
w = sp.Symbol('omega', real=True)

# Definirea functiei necunoscute Q(t)
Q = sp.Function('Q')(t)

# Derivata prima si a doua a functiei Q
dQ_dt = sp.Derivative(Q, t)
d2Q_dt = sp.Derivative(dQ_dt, t)

# Ecuația diferențială
ecuatie = sp.Eq(L * d2Q_dt + R * dQ_dt + (1 / C) * Q, V_0 * sp.cos(w * t))

# Condițiile inițiale: Q(0) = 0 și dQ/dt(0) = 0
conditii_initiale = {Q.subs(t, 0): 0, dQ_dt.subs(t, 0): 0}

# Rezolvarea ecuației diferențiale cu condiții inițiale
sol = sp.dsolve(ecuatie, Q, ics=conditii_initiale)

# Înlocuirea valorilor constantei pentru a evalua soluția
sol_val = sol.subs({L: 10, R: 4, C: 0.1, V_0: 10, w: 5})

# Afișarea soluției
print(sol_val)


#Extragere functie numerica pentru toate valorile lui t 
Q_numeric = sp.lambdify(t, sol_val.rhs, 'numpy')

t_values = np.linspace(0,10, 300)
Q_values = Q_numeric(t_values)


plt.plot(t_values, Q_values, label = 'Evolutia lui Q')
plt.xlabel('Timp')
plt.ylabel('Coulombi')
plt.legend()
plt.grid()
plt.show()


