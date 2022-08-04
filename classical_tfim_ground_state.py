from qmpsyc.classical_opt_tenpy import DMRG_TFIM_infinite, TFIM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xmps.iOptimize import find_ground_state
from xmps.iMPS import iMPS
from xmps.iTDVP import Trajectory
import scipy.integrate as integrate

def infinite_gs_energy(J, g):
    """For comparison: Calculate groundstate energy density of an infinite system.
    The analytic formula stems from mapping the model to free fermions, see
    P. Pfeuty, The one-dimensional Ising model with a transverse field},
    Annals of Physics 57, p. 79 (1970).
    Note that we use Pauli matrices compared this reference using spin-1/2 matrices
    and replace the sum_k -> integral dk/2pi to obtain the result in the N-> infinity limit.
    """
    def f(k, lambda_):
        return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    E0_exact = - g/(J*2.*np.pi) * integrate.quad(f, -np.pi, np.pi, args=(J/g, ))[0]
    return E0_exact


if __name__ == '__main__':

    run = True
    df = pd.read_csv('/home/jamie/Dropbox/QmpSyc/qmpsyc/data/ground_state/classical_gs/reference_gs_energy.csv')
    gs = df['g'].to_list()
    print(len(gs))
    gs_dmrg = np.linspace(gs[0],gs[-1],20)
    gs_dmrg30 = np.linspace(gs[0],gs[-1],30)


    if run:
        J = -1
        Es = []
        Es_tdvp=[]
        Exact_Es = []
        for g in gs:
            #E = DMRG_TFIM_infinite(J=-1, g=g, h=0., Dmax=2)
            #Es.append(E)
            # tfim_H = TFIM(g)
            
            # mps = iMPS().random(d=2,D=2).left_canonicalise()
            # iT = -1j*np.linspace(0, 10, 5000)
            # traj = Trajectory(mps, H=[tfim_H]).eulerint(iT)
            
            # E = traj.mps.left_canonicalise().energy([tfim_H])
            # Es_tdvp.append(E)
            Exact_Es.append(infinite_gs_energy(-J,g))
            # print(f'g={g}')
            # print(f'E={E}')
        #np.save("dmrg_D2_tfim.npy", Es)
        # np.save("tdvp_D2_tfim_gs.npy", Es_tdvp)
        np.save("tfim_exact_gs.npy", Exact_Es)

    plt.plot(gs, Exact_Es, label = "Exact")

    plt.plot(gs, df['Energy'], label = "Circuit")
    plt.legend()    
    plt.show()