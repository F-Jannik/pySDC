from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
import dolfin as df
import numpy as np

c_nvars = 8
family = 'CG'
order = 1

mesh_coarse = df.UnitIntervalMesh(c_nvars)
V_coarse = df.FunctionSpace(mesh_coarse, family, order)

mesh_fine = df.UnitIntervalMesh(2*c_nvars)
V_fine = df.FunctionSpace(mesh_fine, family, order)

points_coarse = fenics_mesh(V_coarse)
for i in range(1,len(points_coarse.values.vector())):
    points_coarse.values.vector()[i] = points_coarse.values.vector()[i-1]+i**2

points_fine = fenics_mesh(V_fine)
for i in range(1, len(points_fine.values.vector())):
    points_fine.values.vector()[i] = points_fine.values.vector()[i-1]+i**2

project = df.project(points_coarse.values, V_fine)
interpolate = df.interpolate(points_coarse.values, V_fine)
print(points_coarse.values.vector()[:])
print(f"Project:\n{project.vector()[:]}")
print(f"Interpolate:\n{interpolate.vector()[:]}")

print("\n\n")

project = df.project(points_fine.values, V_coarse)
interpolate = df.interpolate(points_fine.values, V_coarse)
print(points_fine.values.vector()[:])
print(f"Project:\n{project.vector()[:]}")
print(f"Interpolate:\n{interpolate.vector()[:]}")
