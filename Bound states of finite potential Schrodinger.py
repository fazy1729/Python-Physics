import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Parametrii sistemului
a = 1
hbar = 1
m = 1
z0 = 2*np.pi
V0 = (z0**2 * hbar**2)/(2*m*a**2)

# Definirea ecuațiilor Schrödinger
def ecuatie1(x, y, E):
    """Ecuatia Schrödinger pentru regiunea 1 (0 < x < a)."""
    psi = y[0]
    dpsi_dx = y[1]
    k2 = 2 * m * E / hbar**2
    d2psi_dx2 = -k2 * psi
    return [dpsi_dx, d2psi_dx2]

def ecuatie2(x, y, E):
    """Ecuatia Schrödinger pentru regiunea 2 (a < x < 2a)."""
    psi = y[0]
    dpsi_dx = y[1]
    kappa2 = 2 * m * (V0 - E) / hbar**2
    d2psi_dx2 = kappa2 * psi
    return [dpsi_dx, d2psi_dx2]

# Condiții inițiale
psi0 = 1  # psi(0) = 1
dpsi0_dx = 0  # psi'(0) = 0
y0 = [psi0, dpsi0_dx]

# Funcția care rezolvă ecuațiile pentru un anumit E
def gaseste_E(E):
    # Rezolvăm ecuația pentru regiunea 1 (0 < x < a)
    sol1 = solve_ivp(lambda x, y: ecuatie1(x, y, E), [0, a], y0, method='RK45', t_eval=np.linspace(0, a, 100))
    
    # Valorile psi și dpsi_dx pentru x = a
    psi_a = sol1.y[0][-1]
    dpsi_a = sol1.y[1][-1]
    
    # Condiții inițiale pentru regiunea 2 (a < x < 2a)
    y0_psi2 = [psi_a, dpsi_a]
    
    # Rezolvăm ecuația pentru regiunea 2 (a < x < 2a)
    sol2 = solve_ivp(lambda x, y: ecuatie2(x, y, E), [a, 2*a], y0_psi2, method='RK45', t_eval=np.linspace(a, 2*a, 100))
    
    # Returnăm valoarea lui psi la x = 2a pentru a verifica
    return sol2.y[0][-1]

# Găsim toate stările legate
def gaseste_toate_starile_legate():
    E_values = np.linspace(0.01, 10*V0 - 0.01, 1000)  # Mărim rezoluția pentru precizie
    roots = []

    for i in range(len(E_values) - 1):
        E_start = E_values[i]
        E_end = E_values[i + 1]
        
        if gaseste_E(E_start) * gaseste_E(E_end) < 0:
            rezultat = root_scalar(gaseste_E, bracket=[E_start, E_end], method='brentq')
            roots.append(rezultat.root)
    
    return roots

# Găsim toate stările legate
stari_legate = gaseste_toate_starile_legate()
print("Energiile stărilor legate sunt:", stari_legate)

# Rezolvăm și afișăm soluțiile pentru fiecare stare legată
for E in stari_legate:
    sol1 = solve_ivp(lambda x, y: ecuatie1(x, y, E), [0, a], y0, method='RK45', t_eval=np.linspace(0, a, 100))
    psi_a = sol1.y[0][-1]
    dpsi_a = sol1.y[1][-1]
    
    y0_psi2 = [psi_a, dpsi_a]
    sol2 = solve_ivp(lambda x, y: ecuatie2(x, y, E), [a, 2*a], y0_psi2, method='RK45', t_eval=np.linspace(a, 2*a, 100))
    
    plt.plot(sol1.t, sol1.y[0], label=f'psi (E = {E:.5f}) pe [0, a]')
    plt.plot(sol2.t, sol2.y[0], label=f'psi (E = {E:.5f}) pe [a, 2a]')

plt.grid(True)
plt.xlabel('x')
plt.ylabel('psi(x)')
plt.title('Funcțiile de undă pentru stările legate')
plt.show()