import numpy as np
from scipy.linalg import eig
import scipy.sparse.linalg.eigen.arpack as arp
import cirq
from ncon import ncon
from circuits import StateAnsatzRepeatedXZ

def classically_find_env(U_params,method='Nelder-Mead', testing=False):
    '''
    Find env params classically
    '''
    from tracedistance import TraceDistanceAnalytic
    from functools import partial
    from scipy.optimize import minimize

    Q = cirq.LineQubit.range(4)

    initial_guess = U_params

    cf = lambda x : TraceDistanceAnalytic(Th=U_params, Psi=x, Q=Q)

    env_params = minimize(
        cf,
        method = method,
        x0=initial_guess,
        tol=1e-15,
        options={
            'ftol':1e-15
        }
    )

    if env_params['success'] == False:
        print('Failed to converge on env')

    if testing:
        return env_params

    return env_params['x']


#############################################################################
# Find the exact overlap classically.
#############################################################################

# Note that tensors by default have the shape
# A.shape = (i, j, k)
#
# i -- A -- K
#      |
#      j

def unitary_to_tensor(U):
    '''
    Take a unitary U and make it a tensor A such that

    j ---|     |=== k
         |  U  |
    i ===|     |---|0>

    A.shape = (i, j, k)

    i == A == k
         |
         j
    '''
    n = int(np.log2(U.shape[0]))
    zero = np.array([1., 0.])

    Ucontr = list(range(-1, -2*n, -1)) + [1] # contraction string for ncon
    A = ncon([U.reshape(*2 * n * [2]), zero], [Ucontr, [1,]])
    return A.reshape(2**(n-1), 2, 2**(n-1))


def unitary_to_tensor_v2(U):
    '''
    Take a unitary U and make it a tensor A such that
   |0>     k
    |     |
    |     |     |
    ---U---     | direction of unitary
    |     |     |
    |     |     v
    i     j

    A.shape = (i, j, k)

    i == A == k
         |
         j
    '''
    n = int(np.log2(U.shape[0]))
    zero = np.array([1., 0.])

    Ucontr = list(range(-1, -n-1, -1)) + [1] + list(range(-n-1, -2*n, -1))
    A = ncon([U.reshape(*2 * n *[2]), zero], [Ucontr, [1,]])
    A = A.reshape(2**(n-1), 2, 2**(n-1))
    return A



def map_AB(A, B):
    '''
    Combine A, B as follows

    i -- A -- j    ,   k -- B -- l
         |                  |
    =
    i -- A -- j
         |
    k -- B -- l

    where the shape of the output is (i*k, j*l)
    '''
    i, _, j = A.shape
    k, _, l = B.shape
    return np.einsum('inj, knl -> ikjl', A, B.conj()).reshape(i*k, j*l)


def right_fixed_point(E, all_evals=False):
    '''
    Calculate the right fixed point of a transfer matrix E

    E.shape = (N, N)
    '''
    evals, evecs = eig(E)
    sort = sorted(zip(evals, evecs), key=lambda x: np.linalg.norm(x[0]),
                   reverse=True)
    # Look into `scipy.sparse.linalg.eigs, may be faster`
    if all_evals:
        mu, r = list(zip(*sort))
        return np.array(mu), np.array(r)
    mu, r = sort[0]
    return mu, r


def state_params_to_unitary(θ):
    '''
    Convert state params for qubit to two qubit unitary.
    '''
    from circuits import StateAnsatz
    Q = cirq.LineQubit.range(2)

    return cirq.unitary(StateAnsatz(θ).on(*Q))


def exact_overlap(θ_A, θ_B, ab=True, all_evals=False, Ansatz=StateAnsatzRepeatedXZ):
    '''
    Calculate the overlap classically i.e. abs of max eigenvalue of transfer
    matrix
    '''

    Q = cirq.LineQubit.range(2)

    U = cirq.unitary(
        Ansatz(θ_A).on(*Q)  # 2 qubit circuit
    )

    U_prime = cirq.unitary(
        Ansatz(θ_B).on(*Q)
    )

    A = unitary_to_tensor_v2(U)
    B = unitary_to_tensor_v2(U_prime)

    E = map_AB(A, B)

    mu, r = right_fixed_point(E, all_evals=all_evals)

    if ab:
        return np.abs(mu)
    return mu

def exact_overlap_sigmaZ(θ_A, θ_B, Ansatz=StateAnsatzRepeatedXZ):
    Q = cirq.LineQubit.range(2)

    U = cirq.unitary(
        Ansatz(θ_A).on(*Q)  # 2 qubit circuit
    )

    U_prime = cirq.unitary(
        Ansatz(θ_B).on(*Q)
    )

    A = unitary_to_tensor_v2(U)
    B = unitary_to_tensor_v2(U_prime)

    '''
    A.shape = (i, j, k)

    i -- A -- k
         |
         j
    '''

    I = np.eye(2, 2)

    sigma_z = np.array([[1., 0.],
                       [0., -1.]])

    E = ncon([A, np.conj(A)], [[-1, 1, -3], [-2, 1, -4]])
    Eprime = ncon([A, np.conj(A)], [[-2, 1, -4], [-1, 1, -3]])
    E = E.reshape(4, 4)
    Eprime = Eprime.reshape(4, 4)
    Lambda, R = arp.eigs(E, k=1, which='LM')
    R = R.reshape([2, 2])
    trR = np.trace(R)
    R = R / trR

    sigz = ncon([I, A, sigma_z, np.conj(A), R], [[1, 4], [1, 2, 3], [2,5], [4, 5, 6], [6, 3]])

    return sigz

def exact_overlap_sigmaX(θ_A, θ_B, ab=True, all_evals=False, Ansatz=StateAnsatzRepeatedXZ):
    Q = cirq.LineQubit.range(2)

    U = cirq.unitary(
        Ansatz(θ_A).on(*Q)  # 2 qubit circuit
    )

    U_prime = cirq.unitary(
        Ansatz(θ_B).on(*Q)
    )

    A = unitary_to_tensor_v2(U)
    B = unitary_to_tensor_v2(U_prime)

    I = np.eye(2, 2)

    sigma_x = np.array([[0., 1.],
                        [1., 0.]])


    E = ncon([A, B], [[-1, 1, -3], [-2, 1, -4]])
    E = E.reshape(4, 4)
    Lambda, R = arp.eigs(E, k=1, which='LM')
    R = R.reshape(2, 2)
    trR = np.trace(R)
    R = R / trR
    sigx = ncon([I, A, sigma_x, np.conj(B), R], [[1, 4], [1, 2, 3], [2,5], [4, 5, 6], [3, 6]])

    return sigx

def exact_overlap_sigmaY(θ_A, θ_B, ab=True, all_evals=False, Ansatz=StateAnsatzRepeatedXZ):
    Q = cirq.LineQubit.range(2)

    U = cirq.unitary(
        Ansatz(θ_A).on(*Q)  # 2 qubit circuit
    )

    U_prime = cirq.unitary(
        Ansatz(θ_B).on(*Q)
    )

    A = unitary_to_tensor_v2(U)
    B = unitary_to_tensor_v2(U_prime)

    I = np.eye(2, 2)

    sigma_y = np.array([[0., -1.j],
                        [1.j, 0.]])

    E = ncon([A, B], [[-1, 1, -3], [-2, 1, -4]])
    E = E.reshape(4, 4)
    Lambda, R = arp.eigs(E, k=1, which='LM')
    R = R.reshape(2, 2)
    trR = np.trace(R)
    R = R / trR
    sigy = ncon([I, A, sigma_y, np.conj(B), R], [[1, 4], [1, 2, 3], [2,5], [4, 5, 6], [3, 6]])

    return sigy

def merge(A, B): # TODO Write test for this
    '''
    Merge tensors A, B such that

    -- A -- B --
       |    |
    '''
    ai, aj, ak = A.shape
    bi, bj, bk = B.shape
    return np.einsum('ijk, klm', A, B).reshape(ai, aj*bj, bk)

def exact_n_overlaps(θ_A, θ_B, n_range, ab=True, all_evals=False):
    from circuits import StateAnsatz

    Q = cirq.LineQubit.range(2)

    U = cirq.unitary(
        StateAnsatz(θ_A).on(Q[-1], Q[-2])  # 2 qubit circuit
    )

    U_prime = cirq.unitary(
        StateAnsatz(θ_B).on(Q[-1], Q[-2])
    )

    A = unitary_to_tensor(U)
    B = unitary_to_tensor(U_prime)

    for n in n_range:
        pass

if __name__=="__main__":
    from circuits import ShallowFullStateAnsatz

    np.random.seed(0)
    p0 = np.random.randn(15)
    p1 = np.random.randn(15)
    print('p0: ', p0)
    print('p1: ', p1)

    ov = exact_overlap(p0, p1, Ansatz=ShallowFullStateAnsatz) ** 2
    print(ov)
