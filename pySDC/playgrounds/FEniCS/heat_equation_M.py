import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass, fenics_heat_mass_timebc
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced_jannik import fenics_heat_mass as fenics_heat_mass_test
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
# from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced_jannik import fenics_heat as fenics_heat_test
from pySDC.implementations.problem_classes.Nonlinear_DGL_jannik import fenics_heat as fenics_heat_test
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper

order = 3

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def run_simulation(ml=None, mass=None, dt=0.2, maxiter=20):

    t0 = 0
    dt = 0.2
    Tend = 0.6

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter #20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]

    problem_params = dict()
    problem_params['nu'] = 1.
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [128]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['c'] = 1.0
    if ml:
        problem_params['refinements'] = [1, 0]
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


def visualize(args):


    plt_helper.setup_mpl()

    # plt_helper.newfig(240, 1, ratio=0.8)
    #
    # plt_helper.plt.loglog(
    #     [err[0] for err in errors_sdc_noM],
    #     [err[1] for err in errors_sdc_noM],
    #     lw=2,
    #     marker='s',
    #     markersize=6,
    #     color='darkblue',
    #     label='SDC without M',
    # )
    #
    # plt_helper.plt.xlim([0, 11])
    # plt_helper.plt.ylim([6e-09, 2e-03])
    # plt_helper.plt.xlabel('iteration')
    # plt_helper.plt.ylabel('error')
    # plt_helper.plt.legend()
    # plt_helper.plt.grid()
    #
    # plt_helper.savefig('error_SDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    i = 0
    if "sdc_noM" in args:
        errors_sdc_noM = np.load('npy/errors_sdc_noM.npy')
        err0 = errors_sdc_noM[0][0]
        for err in errors_sdc_noM:
            if err[0]==err0:
                i+=1

        plt_helper.plt.loglog(
            [err[0] for err in errors_sdc_noM],
            [err[1] for err in errors_sdc_noM],
            lw=2,
            color='darkblue',
            marker='s',
            markersize=6,
            label='SDC without M',
        )
        order = (errors_sdc_noM[0][1]/errors_sdc_noM[0][0]) / (errors_sdc_noM[i][1]/errors_sdc_noM[i][0])
        order = max(1,order)
        # plt_helper.plt.plot([errors_sdc_noM[0][0], errors_sdc_noM[-1][0]], [errors_sdc_noM[0][1], errors_sdc_noM[0][1] / (order ** (len(errors_sdc_noM)//i-1))], label=f"{order=}")
    if "sdc_M" in args:
        errors_sdc_M = np.load('npy/errors_sdc_M.npy')
        plt_helper.plt.loglog(
            [err[0] for err in errors_sdc_M],
            [err[1] for err in errors_sdc_M],
            lw=2,
            marker='o',
            markersize=6,
            color='red',
            label='SDC with M',
        )
        order = (errors_sdc_M[0][1]/errors_sdc_M[0][0]) / (errors_sdc_M[i][1]/errors_sdc_M[i][0])
        order = max(1,order)
        # plt_helper.plt.plot([errors_sdc_M[0][0], errors_sdc_M[-1][0]], [errors_sdc_M[0][1], errors_sdc_M[0][1] / (order ** (len(errors_sdc_M)//i-1))], label=f"{order=}")
    if "sdc_noM" in args or "sdc_M" in args:
        # plt_helper.plt.xlim([1e-11, 11])
        # plt_helper.plt.ylim([6e-09, 2e-03])
        plt_helper.plt.xlabel('iteration')
        plt_helper.plt.ylabel('error')
        plt_helper.plt.legend()
        plt_helper.plt.grid()

        plt_helper.savefig('error_SDC_M_CG_4')

    # plt_helper.newfig(240, 1, ratio=0.8)
    #
    # plt_helper.plt.loglog(
    #     [err[0] for err in errors_mlsdc_noM],
    #     [err[1] for err in errors_mlsdc_noM],
    #     lw=2,
    #     marker='s',
    #     markersize=6,
    #     color='darkblue',
    #     label='MLSDC without M',
    # )
    #
    # plt_helper.plt.xlim([0, 11])
    # plt_helper.plt.ylim([6e-09, 2e-03])
    # plt_helper.plt.xlabel('iteration')
    # plt_helper.plt.ylabel('error')
    # plt_helper.plt.legend()
    # plt_helper.plt.grid()
    #
    # plt_helper.savefig('error_MLSDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    if "mlsdc_noM" in args:
        errors_mlsdc_noM = np.load('npy/errors_mlsdc_noM.npy')
        plt_helper.plt.loglog(
            [err[0] for err in errors_mlsdc_noM],
            [err[1] for err in errors_mlsdc_noM],
            lw=2,
            color='darkblue',
            marker='s',
            markersize=6,
            label='MLSDC without M',
        )
        order = (errors_mlsdc_noM[0][1]/errors_mlsdc_noM[0][0]) / (errors_mlsdc_noM[i][1]/errors_mlsdc_noM[i][0])
        order = max(1,order)
        # plt_helper.plt.plot([errors_mlsdc_noM[0][0], errors_mlsdc_noM[-1][0]], [errors_mlsdc_noM[0][1], errors_mlsdc_noM[0][1] / (order ** (len(errors_mlsdc_noM)//i-1))], label=f"{order=}")
    if "mlsdc_M" in args:
        errors_mlsdc_M = np.load('npy/errors_mlsdc_M.npy')
        plt_helper.plt.loglog(
            [err[0] for err in errors_mlsdc_M],
            [err[1] for err in errors_mlsdc_M],
            lw=2,
            marker='o',
            markersize=6,
            color='red',
            label='MLSDC with M',
        )
        order = (errors_mlsdc_M[0][1]/errors_mlsdc_M[0][0]) / (errors_mlsdc_M[i][1]/errors_mlsdc_M[i][0])
        order = max(1,order)
        # plt_helper.plt.plot([errors_mlsdc_M[0][0], errors_mlsdc_M[-1][0]], [errors_mlsdc_M[0][1], errors_mlsdc_M[0][1] / (order ** (len(errors_mlsdc_M)//i-1))], label=f"{order=}")

    if "mlsdc_noM" in args or "mlsdc_M" in args:
        # plt_helper.plt.xlim([1e-11, 11])
        # plt_helper.plt.ylim([6e-09, 2e-03])
        plt_helper.plt.xlabel('iteration')
        plt_helper.plt.ylabel('error')
        plt_helper.plt.legend()
        plt_helper.plt.grid()

        plt_helper.savefig('error_MLSDC_M_CG_4')


if __name__ == "__main__":

    # errors_sdc_noM, _, _ = run_simulation(ml=False, mass=False)
    # errors_sdc_M, _, _ = run_simulation(ml=False, mass=True)
    # errors_mlsdc_noM, _, _ = run_simulation(ml=True, mass=False)
    # errors_mlsdc_M, _, _ = run_simulation(ml=True, mass=True)
    #
    # # print(len(errors_mlsdc_M))
    # # print(len(errors_mlsdc_noM))
    #
    # np.save('npy/errors_sdc_M.npy',  errors_sdc_M)
    # np.save('npy/errors_sdc_noM.npy',  errors_sdc_noM)
    # np.save('npy/errors_mlsdc_M.npy',  errors_mlsdc_M)
    # np.save('npy/errors_mlsdc_noM.npy',  errors_mlsdc_noM)
    #
    # visualize(["mlsdc_M", "mlsdc_noM", "sdc_M", "sdc_noM"])

    error = [[],[],[],[],[],[]]
    u = [[],[],[],[],[],[]]
    for i, maxiter in enumerate([2,4,6,8,10]):
        for dt in [0.2, 0.1, 0.05, 0.025, 0.0125]:
            errors, _, uend = run_simulation(dt=dt, maxiter=maxiter, ml=True, mass=False)
            print(errors)
            error[i].append([dt, errors[-1][1]])
            u[i].append([dt, uend])
    maxiter = 5
    for dt in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        dt *= 0.01
        errors, _, uend = run_simulation(dt=dt, maxiter=maxiter)
        error[5].append([dt*100,errors[-1][1]])
        u[5].append([dt, uend])

    for i, um in enumerate(u[:-1]):
        for j, un in enumerate(um):
            error[i][j][1] = np.linalg.norm(np.array(un[1].values.vector()[:])-np.array(u[-1][j][1].values.vector()[:]))


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
