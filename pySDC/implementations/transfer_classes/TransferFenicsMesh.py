import dolfin as df

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


class mesh_to_mesh_fenics(SpaceTransfer):
    """
    This implementation can restrict and prolong between fenics meshes
    """

    def project(self, F):
        """
        Restriction implementation via projection

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(df.project(F.values, self.coarse_prob.init))
            #print(f"coarseOLD: {u_coarse.values.vector()[:]}")

            u_coarse.values.vector()[0] = F.values.vector()[0] + 0.5 * F.values.vector()[1]
            n = len(u_coarse.values.vector())
            for i in range(1,n-1):
                u_coarse.values.vector()[i] = 0.5 * F.values.vector()[2*i-1] + F.values.vector()[2*i] + 0.5 * F.values.vector()[2*i+1]
            u_coarse.values.vector()[n-1] = 0.5 * F.values.vector()[2*n-3] + F.values.vector()[2*n-2]
            print(f"coarseNEW: {u_coarse.values.vector()[:]}")
            print(f"fine: {F.values.vector()[:]}")

        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = df.project(F.impl.values, self.coarse_prob.init)
            u_coarse.expl.values = df.project(F.expl.values, self.coarse_prob.init)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(df.interpolate(F.values, self.coarse_prob.init))
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = df.interpolate(F.impl.values, self.coarse_prob.init)
            u_coarse.expl.values = df.interpolate(F.expl.values, self.coarse_prob.init)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, fenics_mesh):
            u_fine = fenics_mesh(df.interpolate(G.values, self.fine_prob.init))
        elif isinstance(G, rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.fine_prob.init)
            u_fine.impl.values = df.interpolate(G.impl.values, self.fine_prob.init)
            u_fine.expl.values = df.interpolate(G.expl.values, self.fine_prob.init)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
