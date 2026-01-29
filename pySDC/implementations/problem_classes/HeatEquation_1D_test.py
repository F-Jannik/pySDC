import logging

import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh

start = -1.
end = 1.

class numpy_heat(Problem):
    
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, c_nvars=128, t0=0.0, family = "CG", order=1, refinements=0, nu=1., c=0.0):

        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        for _ in range(refinements):
            c_nvars *= 2

        self.dx = 1/c_nvars
        space_mesh = np.linspace(start,end,c_nvars) #TODO richtigen Start und Endwert einsetzen

        self._makeAttributeAndRegister('c_nvars', 't0', 'family', 'order', 'refinements', 'nu', 'c', localVars=locals(), readOnly=True)
        self.nvars = c_nvars

        self.M = create_mass_matrix(start,end,(c_nvars-1)/(end-start)) #TODO richtigen Start und Endwert einsetzen
        self.K = -create_stiffness_matrix(start,end,(c_nvars-1)/(end-start)) #TODO richtigen Start und Endwert einsetzen

        super().__init__(space_mesh)

        self.bc = bc

        self.g = g

        self.fix_bc_for_residual = True

    def solve_system(self, rhs, factor, u0, t):
        b = self.M @ rhs
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        self.bc(T, b, t)
        u = np.linalg.solve(T, b)

        return u

    def __eval_fexpl(self, u, t):
        x = np.linspace(start,end,self.c_nvars)
        fexpl = self.dtype_u(self.g(x,t)) #TODO M rausnehmen und dafuer in fimpl
        return fexpl

    def __eval_fimpl(self, u ,t):
        return np.linalg.inv(self.M) @ self.K @ u #TODO Mit M_invers multiplizieren

    def eval_f(self, u, t):
        f = self.dtype_f(np.linspace(start,end,self.c_nvars))
        f.impl = self.__eval_fimpl(u, t) #TODO muss der Type gewechselt werden

        f.expl = self.__eval_fexpl(u,t)
        return f

    def apply_mass_matrix(self, u):
        return self.M @ u

    def u_exact(self, t):
        return exact(self.c_nvars, t)

    def fix_residual(self, res):
        self.bc(RHS=res)


def create_stiffness_matrix(start, end, gradient):
    interval_length = 1/gradient
    intervals = int((end-start)/interval_length)+1
    stiffness_matrix = [[0]*intervals for _ in range(intervals)]
    for i in range(intervals):
        stiffness_matrix[i][i] = gradient*2
        if i<intervals-1:
            stiffness_matrix[i][i+1] = -gradient
        if i>0:
            stiffness_matrix[i][i-1] = -gradient
    stiffness_matrix[0][0] = stiffness_matrix[0][0]*0.5
    stiffness_matrix[intervals-1][intervals-1] = stiffness_matrix[intervals-1][intervals-1]*0.5
    return np.array(stiffness_matrix)

def create_mass_matrix(start, end, gradient):
    interval_length = 1/gradient
    intervals = int((end-start)/interval_length)+1
    mass_matrix = [[0]*intervals for _ in range(intervals)]
    for i in range(intervals):
        mass_matrix[i][i] = (4/6) / gradient
        if i<intervals-1:
            mass_matrix[i][i+1] = (1/6) / gradient
        if i>0:
            mass_matrix[i][i-1] = (1/6) / gradient
    mass_matrix[0][0] = mass_matrix[0][0]*0.5
    mass_matrix[intervals-1][intervals-1] = mass_matrix[intervals-1][intervals-1]*0.5
    return np.array(mass_matrix)

def g(x, t):
    return -np.sin(np.pi*x) * (np.sin(t) - np.pi * np.pi * np.cos(t))

def exact(nx, t):
    x = np.linspace(start, end, nx) #TODO richtigen start und endwert einsetzen
    return np.sin(np.pi * x) * np.cos(t)

def bc(LHS=np.array([[0]]), RHS=np.array([0]), t=0):
    sol = exact(2, t)
    RHS[0] = sol[0]
    RHS[len(RHS)-1] = sol[1]
    LHS[0,:] = LHS[len(LHS)-1,:] = 0.
    LHS[0,0] = LHS[len(LHS)-1, len(LHS)-1] = 1.




class numpy_heat_mass(numpy_heat):

    def __init__(self, c_nvars=128, t0=0.0, family = "CG", order=1, refinements=0, nu=1., c=0.0):
        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

    def solve_system(self, rhs, factor, u0, t):
        b = rhs # self.M @ rhs
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        self.bc(T, b, t)
        u = np.linalg.solve(T, b)

        return u

    def eval_f(self, u, t):
        f = self.dtype_f(np.linspace(start,end,self.c_nvars))
        f.impl = self.K @ u

        x = np.linspace(start,end,self.c_nvars)
        f.expl = self.M @ self.dtype_u(self.g(x,t)) #TODO M rausnehmen und dafuer in fimpl
        return f
