from dolfin import *
import numpy as np
mesh = Mesh("sphere2.xml")
Q = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(Q)
v = TestFunction(Q)

class DirichletBoundary(SubDomain):
    def inside (self, x, boundary):
        return boundary

# Optimal parameters
# rho = 42.84661778
# R =  0.5140653
# r = 0.27563518
# T = 50

rho = 10
R = 0.5
r = 0.2
T = 20
alpha = 0.01
theta = 0.5
h = mesh.hmin ()
dt = h
mass = []

# +x[2]*x[2] 3D term
data = Expression("pow(R-sqrt(x[0]*x[0]+x[1]*x[1]), 2)+x[2]*x[2]<=r*r?rho:0.0", degree=1, R=R, r=r, rho=rho)
u0 = Function(Q)
u0 = interpolate(data, Q)
u_init = Function(Q)
u_init = interpolate(data, Q)

g = Constant(0.0)
bc = DirichletBC(Q, g, DirichletBoundary())

a = inner(u, v)*dx + dt * theta * alpha * inner(grad(u), grad(v))*dx
L = inner(u0, v)*dx - dt*(1.0 - theta)*alpha*inner(grad(u0), grad(v))*dx
A = assemble(a)
b = assemble(L)
bc.apply(A, b)

sol = Function(Q)
M = (u_init-u0)*dx
t = 0.0

file = File("Resultc1/Sphere.pvd")
while t<T:
    file << u0
    assemble(M)
    mass.append(assemble(M))
    solve(A, sol.vector(), b)
    u0.assign(sol)
    b = assemble(L)
    bc.apply(b)
    t += dt
