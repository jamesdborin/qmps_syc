import cirq
import numpy as np
from numpy.lib.function_base import append

###################################
# QMPS Circuits
###################################

class DiagonalEnvironmentAnsatz(cirq.Gate):
    def __init__(self, Th):
        self.Th = Th

    def _decompose_(self, qubits):
        return [
                cirq.ry(self.Th).on(qubits[0]),
                cirq.CNOT(qubits[0], qubits[1])
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['V','V']

class SU2Gate(cirq.Gate):
    """Single qubit operator"""
    def __init__(self, p):
        assert len(p) == 3, "SU(2) has 3 generators"
        self.p = p
    def num_qubits(self) -> int:
        return 1
    def _decompose_(self, qubits):

        yield cirq.rz(self.p[0]).on(qubits[0])
        yield cirq.rx(self.p[1]).on(qubits[0])
        yield cirq.rz(self.p[2]).on(qubits[0])

class LocalGate(cirq.Gate):
    """KAK decomposition of a 2 qubit gate with 15 params. Called local because they act within a Thermofield Hilbert space, not between Hilbert Spaces"""
    def __init__(self, p):
        assert len(p) == 15, "SU(4) has 15 generators"
        self.p = p
    def num_qubits(self) -> int:
        return 2
    def _circuit_diagram_info_(self, args):
        return ['L','L']
    def _decompose_(self, qubits):

        yield SU2Gate(self.p[0:3]).on(qubits[0])
        yield SU2Gate(self.p[3:6]).on(qubits[1])
        yield cirq.XXPowGate(exponent=self.p[6]).on(*qubits)
        yield cirq.YYPowGate(exponent=self.p[7]).on(*qubits)
        yield cirq.ZZPowGate(exponent=self.p[8]).on(*qubits)
        yield SU2Gate(self.p[9:12]).on(qubits[0])
        yield SU2Gate(self.p[12:]).on(qubits[1])


class EnvironmentAnsatz(cirq.Gate):
    def __init__(self, Th):
        self.Th = Th

    def _decompose_(self, qubits):
        return [
                cirq.ry(self.Th[0]).on(qubits[0]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.rz(self.Th[1]).on(qubits[1]),
                cirq.ry(self.Th[2]).on(qubits[1]),
                cirq.rz(self.Th[3]).on(qubits[1])
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['V','V']


class StateAnsatz(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.ry(self.Psi[0]).on(qubits[0]),
                cirq.ry(self.Psi[1]).on(qubits[1]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.ry(self.Psi[2]).on(qubits[0]),
                cirq.ry(self.Psi[3]).on(qubits[1])
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']


class D0StateAnsatxXZ(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.rx(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[1]),
                cirq.rx(self.Psi[3]).on(qubits[1]),
                cirq.rz(self.Psi[4]).on(qubits[0]),
                cirq.rx(self.Psi[5]).on(qubits[0]),
                cirq.rz(self.Psi[6]).on(qubits[1]),
                cirq.rx(self.Psi[7]).on(qubits[1]),
#                cirq.rz(self.Psi[8]).on(qubits[0]),
#                cirq.rx(self.Psi[9]).on(qubits[0]),
#                cirq.rz(self.Psi[10]).on(qubits[1]),
#                cirq.rx(self.Psi[11]).on(qubits[1]),
#                cirq.CNOT(*qubits)
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']



class StateAnsatzXZ(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.rx(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[1]),
                cirq.rx(self.Psi[3]).on(qubits[1]),
                cirq.CNOT(*qubits),
                cirq.rz(self.Psi[4]).on(qubits[0]),
                cirq.rx(self.Psi[5]).on(qubits[0]),
                cirq.rz(self.Psi[6]).on(qubits[1]),
                cirq.rx(self.Psi[7]).on(qubits[1]),
                cirq.CNOT(*qubits),
#                cirq.rz(self.Psi[8]).on(qubits[0]),
#                cirq.rx(self.Psi[9]).on(qubits[0]),
#                cirq.rz(self.Psi[10]).on(qubits[1]),
#                cirq.rx(self.Psi[11]).on(qubits[1]),
#                cirq.CNOT(*qubits)
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class StateAnsatzXZ1(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.rx(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[1]),
                cirq.rx(self.Psi[3]).on(qubits[1]),
                cirq.CNOT(*qubits),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class StateAnsatzKak(cirq.Gate):
    def __init__(self, Psi):
        assert len(Psi) == 15
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.ry(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[0]),

                cirq.rz(self.Psi[3]).on(qubits[1]),
                cirq.ry(self.Psi[4]).on(qubits[1]),
                cirq.rz(self.Psi[5]).on(qubits[1]),

                cirq.XX(*qubits)**self.Psi[6],
                cirq.YY(*qubits)**self.Psi[7],
                cirq.ZZ(*qubits)**self.Psi[8],

                cirq.rz(self.Psi[9]).on(qubits[0]),
                cirq.ry(self.Psi[10]).on(qubits[0]),
                cirq.rz(self.Psi[11]).on(qubits[0]),

                cirq.rz(self.Psi[12]).on(qubits[1]),
                cirq.ry(self.Psi[13]).on(qubits[1]),
                cirq.rz(self.Psi[14]).on(qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class ShallowFullStateAnsatz(cirq.Gate):
    def __init__(self, Psi):
        assert len(Psi) == 15
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
            cirq.rz(self.Psi[0]).on(qubits[0]),
            cirq.rx(self.Psi[1]).on(qubits[0]),
            cirq.rz(self.Psi[2]).on(qubits[0]),

            cirq.rz(self.Psi[3]).on(qubits[1]),
            cirq.rx(self.Psi[4]).on(qubits[1]),
            cirq.rz(self.Psi[5]).on(qubits[1]),

            cirq.CNOT(qubits[0], qubits[1]),

            cirq.ry(self.Psi[6]).on(qubits[0]),

            cirq.CNOT(qubits[1], qubits[0]),

            cirq.ry(self.Psi[7]).on(qubits[0]),
            cirq.rz(self.Psi[8]).on(qubits[1]),

            cirq.CNOT(qubits[0], qubits[1]),

            cirq.rz(self.Psi[9]).on(qubits[0]),
            cirq.rx(self.Psi[10]).on(qubits[0]),
            cirq.rz(self.Psi[11]).on(qubits[0]),

            cirq.rz(self.Psi[12]).on(qubits[1]),
            cirq.rx(self.Psi[13]).on(qubits[1]),
            cirq.rz(self.Psi[14]).on(qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']



class StateAnsatzRepeatedXZ(cirq.Gate):
    def __init__(self, ψ, p=3):
        self.ψ = ψ
        self.p = p


    def _decompose_(self, qubits):
        ops = [
                cirq.rz(self.ψ[0]).on(qubits[0]),
                cirq.rx(self.ψ[1]).on(qubits[0]),
                cirq.rz(self.ψ[2]).on(qubits[1]),
                cirq.rx(self.ψ[3]).on(qubits[1]),
                cirq.CNOT(*qubits),
        ] * self.p

        return ops

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class StateAnsatzXY(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rx(self.Psi[0]).on(qubits[0]),
                cirq.ry(self.Psi[1]).on(qubits[1]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.rx(self.Psi[2]).on(qubits[0]),
                cirq.ry(self.Psi[3]).on(qubits[1])
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class ZZMeasure(cirq.Gate):

	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.CNOT(qubits[1], qubits[0])]


class XIMeasure(cirq.Gate):
	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.H(qubits[0])]



class IXMeasure(cirq.Gate):

	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.H(qubits[1])]


## AFM Heinsenberg Hamiltonian
class XXMeasure(cirq.Gate):

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.H(qubits[1]), cirq.H(qubits[2])
            ]

## AFM Heinsenberg Hamiltonian Terms H = ∑ J * (ZZ + YY + XX ) + ∑ g (ZI + IZ)
class YYMeasure(cirq.Gate):

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            cirq.CNOT(qubits[1], qubits[0]),
            cirq.Z(qubits[1])**(-0.5), cirq.Z(qubits[0])**(-0.5),
            cirq.H(qubits[1]), cirq.H(qubits[0])
            ]


class IXMeasureSWAP(cirq.Gate):

	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.SWAP(qubits[0], qubits[1]), cirq.H(qubits[0])]


class ZIMeasure(cirq.Gate):
	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.I(qubits[0])]


class ZIMeasure(cirq.Gate):
	def num_qubits(self):
		return 2

	def _decompose_(self, qubits):
		return [cirq.I(qubits[1])]


def JustEnv(Th, Q):
    # Produce just the environment (Alternative way to get exact trace distance)
    c = cirq.Circuit()
    c.append( cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[-1],Q[-2]])))

    return c


def SSSt(Th, Psi, Q):
    # Produce a single site state with environment (Alternative way to get exact trace distance)
    c = cirq.Circuit()
    c.append( cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[-1],Q[-2]])) )
    c.append( cirq.decompose_once(StateAnsatz(Psi).on(*[Q[-2], Q[-3]])) )

    return c


def EPCircuit(Th,  Q):
    # 4 qubit circuit to calculat the environment purity
    c = cirq.Circuit()
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[0],Q[1]])))
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[3],Q[2]])))

    c.append(cirq.CNOT(Q[1],Q[2]))
    c.append(cirq.H(Q[1]))

    return c


def SPCircuit(Th, Psi, Q):
    # 6 qubit circuit to calculate the environment purity

    c = cirq.Circuit()
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[0], Q[1]])))
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[-1], Q[-2]])))
    c.append(cirq.decompose_once(StateAnsatz(Psi).on(*[Q[1],Q[2]])))
    c.append(cirq.decompose_once(StateAnsatz(Psi).on(*[Q[-2],Q[-3]])))

    c.append(cirq.CNOT(Q[2],Q[3]))
    c.append(cirq.H(Q[2]))

    return c


def SECircuit(Th, Psi, Q):
    # 5 qubit circuit to calculate environment state overlap

    c = cirq.Circuit()
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[0],Q[1]])))

    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[-1],Q[-2]])))
    c.append(cirq.decompose_once(StateAnsatz(Psi).on(*[Q[-2],Q[-3]])))

    c.append(cirq.CNOT(Q[1], Q[2]))
    c.append(cirq.H(Q[1]))

    return c


def StCircuit(Th, Psi, Q):
    # 4 qubit circuit to produce the infinite TI State

    c = cirq.Circuit()
    c.append(cirq.decompose_once(EnvironmentAnsatz(Th).on(*[Q[-1], Q[-2]])))
    c.append(cirq.decompose_once(StateAnsatz(Psi).on(*[Q[-2],Q[-3]])))
    c.append(cirq.decompose_once(StateAnsatz(Psi).on(*[Q[-3],Q[-4]])))

    return c


def NSiteCircuit(Th, Psi, Q, N):
    # build an n-site circuit with a right environment
    c = cirq.Circuit()
    c.append(cirq.decompose_once( EnvironmentAnsatz(Th).on(*[Q[-1], Q[-2]]) ))

    for i in range(N):
        c.append(cirq.decompose_once( StateAnsatz(Psi).on( Q[-2-i], Q[-3-i] ) ))

    return c

def NSiteCircuit_Id_V(Psi, Q, N):
    # build an n-site circuit with right env set to identity
    c = cirq.Circuit()

    for i in range(N):
        c.append(cirq.decompose_once( StateAnsatz(Psi).on( Q[-2-i], Q[-1-i] ) ))

    return c

def NSiteCircuit_Ansatz(Psi, Q, N, Ansatz= StateAnsatzXZ):
    ''' build an n-site circuit with right env set to identity
    use a Kak decomposition for the state ansatz
    '''
    c = cirq.Circuit()

    for i in range(N):
        c.append(cirq.decompose_once( Ansatz(Psi).on( Q[-2-i], Q[-1-i] ) ))

    return c

def NSiteCircuit_Ansatz_Env(Psi, Theta, Q, N, Ne, Ansatz=StateAnsatzXZ, offset=0):

    c = cirq.Circuit()

    for i in range(Ne):
        c.append(cirq.decompose_once( Ansatz(Theta).on( Q[-2-i], Q[-1-i] )))

    for i in range(N):
        c.append(cirq.decompose_once( Ansatz(Psi).on( Q[-2-i-Ne-offset], Q[-1-i-Ne-offset] ) ) )

    return c

def NSiteCircuit_Ansatz_Env0(Psi, Theta, Q, N, Ne, Ansatz=StateAnsatzXZ):

    c = cirq.Circuit()

    for i in range(1, Ne+1):
        c.append(cirq.decompose_once( Ansatz(Theta).on( Q[-2-i], Q[-1-i] )))

    for i in range(1, N+1):
        c.append(cirq.decompose_once( Ansatz(Psi).on( Q[-2-i-Ne], Q[-1-i-Ne] ) ) )

    return c

def NSiteCircuit_AnsatzE(Psi, Q, N, Ansatz= StateAnsatzXZ):
    ''' build an n-site circuit with right env set to identity
    use a Kak decomposition for the state ansatz
    '''
    c = cirq.Circuit()
    ps = len(Psi)
    c.append(cirq.decompose_once( Ansatz(Psi + np.random.rand(ps)).on(Q[-2], Q[-1]) ))

    for i in range(N):
        c.append(cirq.decompose_once( Ansatz(Psi).on( Q[-3-i], Q[-2-i] ) ))

    return c

def apply_U(U, Q):
    '''
    Apply arbitrary 2 qubit unitary U to qubits Q
    '''
    from cirq.optimizers import two_qubit_matrix_to_operations
    return two_qubit_matrix_to_operations(Q[0], Q[1], U, False)


def overlap_circuit(ThA, PsiA, ThB, PsiB, Q, N ):
    cA = NSiteCircuit(ThA, PsiA, Q, N)
    cB = NSiteCircuit(ThB, PsiB, Q, N)

    cA.append(cirq.inverse(cB))

    return cA

def overlap_circuit_V(PsiA, PsiB, Q, N , Ansatz=StateAnsatzXZ):
    cA = NSiteCircuit_Ansatz(PsiA, Q, N, Ansatz)
    cB = NSiteCircuit_Ansatz(PsiB, Q, N, Ansatz)

    cA.append(cirq.inverse(cB))

    return cA

def overlap_circuit_VE(PsiA, PsiB, Q, N , Ansatz=StateAnsatzXZ):
    assert len(Q) == N+2
    cA = NSiteCircuit_AnsatzE(PsiA, Q, N, Ansatz)
    cB = NSiteCircuit_AnsatzE(PsiB, Q, N, Ansatz)

    cA.append(cirq.inverse(cB))

    return cA


def overlap_circuit_Kak(PsiA, PsiB, Q, N ):
    cA = NSiteCircuit_Kak(PsiA, Q, N)
    cB = NSiteCircuit_Kak(PsiB, Q, N)

    cA.append(cirq.inverse(cB))

    return cA


def AddMeasure(circuit, Qs, name):
    # Add measurements to a circuit in the given locations

    circuit.append(cirq.measure(*Qs, key = name), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit


def AddHamTerm(StCircuit, HamTerm, Qs):

    StCircuit.append(cirq.decompose_once(HamTerm().on(*Qs)))
    return StCircuit


#######################################
# Confusion Matrix Circuits
#######################################


# Circuits for two qubit confusion matrix (Trace Distances)
#   Note: Because the trace distance calculation is invariant under 01->10 permuations, we do not
#         need to worry which circuit is the 01 and 10 circuit.
def ZeroZeroCircuit(Q):
    c = cirq.Circuit()
    c.append(cirq.measure(*Q, key = '00'))
    return c


def ZeroOneCircuit(Q):
    c = cirq.Circuit()
    c.append(cirq.X(Q[1]))
    c.append(cirq.measure(*Q, key = '01'))

    return c


def OneZeroCircuit(Q):
    c = cirq.Circuit()
    c.append(cirq.X(Q[0]))
    c.append(cirq.measure(*Q, key = '10'))

    return c

def OneOneCircuit(Q):
    c = cirq.Circuit()
    c.append([cirq.X(Q[0]),cirq.X(Q[1])])
    c.append(cirq.measure(*Q, key = '11'))

    return c

# Circuits for single qubit confusion matrix (Energy calculations need this)
def ZeroCircuit(Q):
    c = cirq.Circuit()
    c.append(cirq.measure(Q, key = '0'))
    return c

def OneCircuit(Q):
    c = cirq.Circuit()
    c.append(cirq.X(Q))
    c.append(cirq.measure(Q, key = '1'))
    return c


class D4RealAnsatz(cirq.Gate):
    def __init__(self, p):

        # 5 params in p
        self.p = p

    def num_qubits(self) -> int:
        return 3

    def _decompose_(self, qubits):
        return [
            cirq.ry(self.p[0]).on(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.ry(self.p[1]).on(qubits[0]),
            cirq.ry(self.p[2]).on(qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.ry(self.p[3]).on(qubits[1]),
            cirq.ry(self.p[4]).on(qubits[2])
        ]

class D4EnvAnsatz(cirq.Gate):
    def __init__(self, p):
        # 8 params in all
        self.p = p

    def num_qubits(self) -> int:
        return 4

    def _decompose_(self, qubits):
        return [
            cirq.ry(self.p[0]).on(qubits[0]),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.ry(self.p[1]).on(qubits[1]),
            cirq.CNOT(qubits[1], qubits[3]),
            cirq.rz(self.p[2]).on(qubits[2]),
            cirq.ry(self.p[3]).on(qubits[2]),
            cirq.rz(self.p[4]).on(qubits[2]),
            cirq.rz(self.p[5]).on(qubits[3]),
            cirq.ry(self.p[6]).on(qubits[3]),
            cirq.rz(self.p[7]).on(qubits[3])
        ]


def D4StateCircuit(state_params, env_params, Q):

    env_q = [Q[-1], Q[-2], Q[-3], Q[-4]]
    state_q1 = [Q[-5], Q[-4], Q[-3]]
    state_q2 = [Q[-6], Q[-5], Q[-4]]

    c = cirq.Circuit()
    c.append(cirq.decompose_once( D4EnvAnsatz( env_params ).on(*env_q)))
    c.append(cirq.decompose_once( D4RealAnsatz( state_params ).on(*state_q1)))
    c.append(cirq.decompose_once( D4RealAnsatz( state_params ).on(*state_q2)))

    return c


def D4SingleSiteCircuit(state_params, env_params, Q):

    env_q = [Q[-1], Q[-2], Q[-3], Q[-4]]
    state_q = [Q[-5], Q[-4], Q[-3]]

    c = cirq.Circuit()
    c.append(cirq.decompose_once( D4EnvAnsatz( env_params ).on(*env_q)))
    c.append(cirq.decompose_once( D4RealAnsatz( state_params ).on(*state_q)))

    return c


def D4JustEnv(env_params, Q):

    env_qs = [Q[-1], Q[-2], Q[-3], Q[-4]]
    c = cirq.Circuit()
    c.append(cirq.decompose_once( D4EnvAnsatz( env_params ).on(*env_qs)))

    return c

def Gate_to_Unitary(params, Ansatz):
    return cirq.unitary(Ansatz(params))


if __name__ == '__main__':
    envParams = np.random.rand(8)
    stateParams = np.random.rand(5)

    Q = cirq.GridQubit.rect(1,6)
    c1 = D4StateCircuit( stateParams, envParams, Q )

    print(c1.to_text_diagram(transpose=True))

    c2 = D4SingleSiteCircuit(stateParams, envParams, Q[:5])

    print('\n\n\n')
    print(c2.to_text_diagram(transpose=True))

    c3 = D4JustEnv(envParams, Q[:4])

    print('\n\n\n')
    print(c3.to_text_diagram(transpose=True))
