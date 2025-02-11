import sympy as sp


r = sp.Symbol('r', real = True, positive = True)
V = sp.Function('V')(r)

dV_dr = sp.Derivative(V, r)
d2V_dr = sp.Derivative(dV_dr, r)


eq = sp.Eq ((1/r**2) * (r**2 * dV_dr).diff(r), (2/(1-4*V) * dV_dr**2))
sol = sp.dsolve(eq, V)
print(sol)
