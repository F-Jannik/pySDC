import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
# from pySDC.implementations.problem_classes.Nonlinear_DGL_jannik import fenics_heat
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import get_list_of_types, get_sorted
import pySDC.helpers.plot_helper as plt_helper

def run_simulation(ml=None, mass=None, dt=0.2, maxiter=20):

    t0 = 0
    # dt = 0.5
    Tend = 1

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter #20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [5]

    problem_params = dict()
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [512]
    problem_params['family'] = 'CG'
    problem_params['order'] = [5]
    if ml:
        problem_params['refinements'] = [1,0]
    else:
        problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()

    # === Das ist mit der nicht invertierten Massematrix
    if mass:
        description['problem_class'] = fenics_heat_mass#fenics_heat_mass_timebc
        description['sweeper_class'] = imex_1st_order_mass
        description['base_transfer_class'] = base_transfer_mass
    # === Das ist mit der invertierten Massematrix
    else:
        description['problem_class'] = fenics_heat
        description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')

    return errors, residuals, uend


def visualize(errors):


    plt_helper.setup_mpl()
    plt_helper.plt.gca().invert_xaxis()

    # plt_helper.newfig(240, 1, ratio=0.8)

    for i, error in enumerate(errors):
        plt_helper.plt.loglog(
            [err[0] for err in error],
            [err[1] for err in error],
            lw=2,
            # color='darkblue',
            marker='s',
            markersize=6,
            label=f"k={i+1}"
        )
    for i, error in enumerate(errors):
        # order = (error[0][1] / error[1][1])-1
        # order = max(1,order)
        order = i+1
        plt_helper.plt.plot([error[0][0], error[-1][0]], [error[0][1], error[0][1] / 2 ** (order*len(error)-1)], label=f"{order=}")
    plt_helper.plt.legend()
    # plt_helper.plt.show()
    plt_helper.savefig('error')

if __name__ == "__main__":

    error = [[],[],[],[],[],[]]
    u = [[],[],[],[],[],[]]
    for maxiter in [1,2,3,4,5]:
        for dt in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
            dt *= 0.5
            errors, _, uend = run_simulation(dt=dt, maxiter=maxiter, ml=False, mass=False)
            error[maxiter-1].append([dt, errors[-1][1]])
            u[maxiter-1].append([dt, uend])

    maxiter = 20
    dt = 0.01 * 0.03125
    errors, _, uend = run_simulation(dt=dt, maxiter=maxiter)
    for _ in range(5):
        error[5].append([dt*100,errors[-1][1]])
        u[5].append([dt, uend])

    # exa = np.array(u[-1][0][1].values.vector()[:])
    # np.save('exact_solution.npy',  exa)

    # exa = np.load("exact_solution.npy")
    # dt = 0.01 * 0.03125
    # for _ in range(5):
    #     u[-1].append([dt*100, exa])

    for i, um in enumerate(u[:-1]):
        for j, un in enumerate(um):
            # error[i][j][1] = np.linalg.norm(np.array(un[1].values.vector()[:])-u[-1][j][1])
            error[i][j][1] = np.linalg.norm(np.array(un[1].values.vector()[:])-np.array(u[-1][j][1].values.vector()[:]))

    # === Just to plot the "exact" solution which is just a smaller step ===
    # error[-1][0] = np.linspace(0,20,len(u[-1][-1][1].values.vector()[:]))
    # error[-1][1] = u[-1][-1][1].values.vector()[:]

    visualize(error[:-1])


    if(False):
        with open("txt/Iter_Errors_noM.txt", "w") as f:

            f.write(str(("k=", "dt", "error")))
            f.write('\n')


            dts = (0.1, 0.05, 0.025, 0.0125)
            maxiters = (1, 2, 3, 4, 5)
            for maxiter in maxiters:
                for dt in dts:
                    errors, _ = run_simulation(ml=True, mass=False, dt=dt, maxiter=maxiter)
                    tuple = (maxiter, dt, errors[-1][-1])
                    f.write(str(tuple))
                    f.write("\n")

        with open("txt/Iter_Errors_M.txt", "w") as f:

            f.write(str(("k=", "dt", "error")))
            f.write('\n')


            dts = (0.1, 0.01, 0.001, 0.0001)
            maxiters = (20,20,20,20,20)
            for maxiter in maxiters:
                for dt in dts:
                    errors, _ = run_simulation(ml=True, mass=True, dt=dt, maxiter=maxiter)
                    tuple = (maxiter, dt, errors[-1][-1])
                    print(f"\n\nErrors:{errors}")
                    print(f"{errors_sdc_noM}\n")
                    f.write(str(tuple))
                    f.write("\n")
