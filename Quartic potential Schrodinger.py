import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Define the potential V(x) = x^4
def potential(u):
    return u**4

# Define the system of ODEs
def schrodinger_equation(u, y, c):
    psi, dpsi_du = y
    d2psi_du2 = 2 * (potential(u) - c) * psi
    return [dpsi_du, d2psi_du2]

# Define the shooting function to find the correct energy eigenvalue
def shooting_function(c):
    # Initial conditions for even solutions: psi(0) = 1, psi'(0) = 0
    y0 = [1, 0]
    # Solve the ODEs
    sol = solve_ivp(lambda u, y: schrodinger_equation(u, y, c), [0, 3.5], y0, t_eval=np.linspace(0, 3.5, 1000))
    # Return the value of psi at u = 3.5
    return sol.y[0][-1]

# Find the energy eigenvalue c using the shooting method
# We need to find c such that psi(3.5) = 0 for even solutions

def all_energies():
    Values_E = np.linspace(0,3.5, 1000)
    radacini = []

    for i in range(len(Values_E)-1):
        E1 = Values_E[i]
        E2 = Values_E[i+1]
        if shooting_function(E1) * shooting_function(E2) < 0:
            rezultat = root_scalar(shooting_function, bracket=[E1,E2], method='brentq')
            radacini.append(rezultat.root)
    return radacini

toate_energiile = all_energies()
print(toate_energiile)

result = root_scalar(shooting_function, bracket=[0.1, 1.0], method='brentq')
c = result.root
print(f"Found energy eigenvalue: {c}")

# Solve the ODEs with the found energy eigenvalue
y0 = [1, 0]
sol = solve_ivp(lambda u, y: schrodinger_equation(u, y, c), [0, 3.5], y0, t_eval=np.linspace(0, 3.5, 1000))

# Plot the solution
plt.plot(sol.t, sol.y[0], label=f'ψ(u) for c = {c:.4f}')
plt.xlabel('u')
plt.ylabel('ψ(u)')
plt.title('Solution of the Schrödinger equation for a quartic potential')
plt.legend()
plt.grid(True)
plt.show()