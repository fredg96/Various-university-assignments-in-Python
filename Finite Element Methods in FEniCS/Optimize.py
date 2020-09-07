from dolfin import *
import scipy.optimize as optimize
from scipy.optimize import minimize
mesh = Mesh("sphere2.xml")
Q = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(Q)
v = TestFunction(Q)

alpha = 0.01
theta = 0.5
dt = 0.1
rho = 45
R = 0.3
r = 0.1
T = 50
data = [rho, R, r]

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def func(data):
    print(data)
    rho = data[0]
    R = data[1]
    r = data[2]
    indata = Expression("pow(R-sqrt(x[0]*x[0] + x[1]*x[1]), 2)+x[2]*x[2] <= r*r?rho:0.0", degree=1, R=R, r=r, rho=rho)
    u0 = Function(Q)
    u0 = interpolate(indata, Q)
    u_init = Function(Q)
    u_init = interpolate(indata, Q)

    g = Constant(0.0)
    bc = DirichletBC(Q, g, DirichletBoundary())

    a = inner(u, v)*dx+dt*theta*alpha*inner(grad(u), grad(v))*dx
    L = inner(u0, v)*dx-dt*(1.0 - theta)*alpha*inner(grad(u0), grad(v))*dx
    M = (u_init-u0)*dx

    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    sol = Function(Q)
    t = dt
    mass = []
    while t < T:
        mass.append(assemble(M))
        solve(A, sol.vector(), b)
        u0.assign(sol)
        b = assemble(L)
        bc.apply(b)
        t += dt
    F = (mass[5*10-1]-10)**2+(mass[7*10-1]-15)**2+(mass[30*10-1]-30)**2
    print(F)
    return F

res = minimize(func, [44, 0.3, 0.1], method='Nelder-Mead', tol=1e-3)
print(res)
