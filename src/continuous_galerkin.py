from firedrake import *
import finat
import matplotlib.pyplot as plt
import math

mesh = PeriodicUnitSquareMesh(50, 50)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

u_np1 = Function(V)  # timestep n+1
u_n = Function(V)    # timestep n
u_nm1 = Function(V)  # timestep n-1


outfile = File("results/continuous_galerkin.pvd")

dt = 0.001
T = 10

t = 0

step = 0

u_0 = sin(2*math.pi*x)
u_m1 = sin(2*math.pi*(x-dt))

u_n.assign(interpolate(u_0, V))
u_nm1.assign(interpolate(u_m1, V))

# trisurf(u_n)
# plt.show()

outfile.write(u_n, time=t)

m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dx

a = dot(grad(u_n), grad(v)) * dx


R = Function(V)
F = m + a 
a, r = lhs(F), rhs(F)
A = assemble(a)
solver = LinearSolver(A, solver_parameters={"ksp_type":"cg"})

step = 0
while t + 1e-6 < T:
    step += 1

    # Update the RHS vector according to the current simulation time `t`

    R = assemble(r, tensor=R)

    # Call the solver object to do point-wise division to solve the system.

    solver.solve(u_np1, R)

    # Exchange the solution at the two time-stepping levels.

    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    # Increment the time and write the solution to the file for visualization in ParaView.

    t += dt
    if step % 100 == 0:
        print("Elapsed time is: "+str(t))
        
outfile.write(u_n, time=t)

trisurf(u_n)
plt.show()