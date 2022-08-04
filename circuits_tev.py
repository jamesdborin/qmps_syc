from cirq.circuits import circuit
from  qmpsyc.circuits import StateAnsatzXZ, NSiteCircuit_Ansatz
from qmpsyc.floquet import ApplyFloquet
from  qmpsyc.circuits import AddMeasure
from qmpsyc.simulate import SimulateCircuitLocalExact, SimulateCircuitGoogle
from qmpsyc.simulate import SimulateGoogleBatched, SimulateCircuitLocalNoisyV

import cirq
import cirq_google as cg
from cirq.optimizers import two_qubit_matrix_to_operations
import numpy as np
from copy import copy

class TFIOverlapCircuitGenerator():
    def __init__(self, U1, U2, n, qubits=None, Ansatz=StateAnsatzXZ):
        self.n = 2*n
        self.qubits = qubits
        self.U1_layer = self._generate_U1(U1)
        self.U2_layer = self._generate_U2(U2)
        self.id_layer = None
        self.Ansatz = Ansatz

        # noisy sim parameters
        self.p = 0.005
        self.shots = 10000
        self.repeats = 1

        # machine run params
        self.processor = 'rainbow'

    def _generate_U1(self, U1):
        ops = []
        for i in range(1, self.n+1, 2):
            q0, q1 = self.qubits[i], self.qubits[i+1]
            ops.extend(two_qubit_matrix_to_operations(q0, q1, U1, True, clean_operations=True))

        return ops

    def _generate_U2(self, U2):
        ops = []
        for i in range(2, self.n+1, 2):
            q0, q1 = self.qubits[i], self.qubits[i+1]
            ops.extend(two_qubit_matrix_to_operations(q0, q1, U2, True, clean_operations=True))

        return ops

    def set_small_dt_layer(self, id_U):
        self.id_layer = self._generate_U2(id_U)

    def set_noisy_params(self, shots=None, reps=None, p=None):
        if shots is not None:
            self.shots = shots

        if reps is not None:
            self.reps = reps

        if p is not None:
            self.p = p

    def overlap_circuit(self, θ_A, θ_B):
        # Prep state circuits
        circuitA = NSiteCircuit_Ansatz(θ_A, self.qubits, self.n+1,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, self.qubits, self.n+1,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U1_layer)
        circuit.append(self.U2_layer)
        circuit.append(self.U1_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def right_env_circuit(self, θ_A, θ_B):
        circuitA = NSiteCircuit_Ansatz(θ_A, self.qubits, 2,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, self.qubits, 2,
                                       Ansatz=self.Ansatz)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def identity_circuit(self, θ):
        assert self.id_layer is not None, 'Need to set identity circuit'
        circuitA = NSiteCircuit_Ansatz(θ, self.qubits, self.n+1,
                                       Ansatz=self.Ansatz)
        circuitA_inv = cirq.inverse(circuitA)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U1_layer)
        circuit.append(self.id_layer)
        circuit.append(cirq.inverse(self.U1_layer))
        circuit.append(circuitA_inv)

        return circuit

    def identity_circuit2(self, θ):
        assert self.id_layer is not None, 'Need to set identity circuit'
        circuitA = NSiteCircuit_Ansatz(θ, self.qubits, self.n+1,
                                       Ansatz=self.Ansatz)
        circuitA_inv = cirq.inverse(circuitA)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U1_layer)
        circuit.append(cirq.inverse(self.U1_layer))
        circuit.append(circuitA_inv)

        return circuit

    def prob0_overlap_exact(self, θ_A, θ_B, circuit=None):
        if circuit==None:
            circuit = self.overlap_circuit(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_overlap_noisy(self, θ_A, θ_B, p=None, shots=None, repeats=None):
        if p is None:
            p = self.p

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        probs = []
        for _ in range(repeats):
            noise = cirq.depolarize(p=p)
            circuit = self.overlap_circuit(θ_A, θ_B)
            circuit = AddMeasure(circuit, self.qubits, 'meas')
            result = SimulateCircuitLocalNoisyV(circuit, shots, noise).histogram(key='meas')

            prob0 = result[0] / sum(result.values())
            probs.append(prob0)

        p0_mean = np.mean(probs)
        p0_std = np.std(probs)
        return p0_mean, p0_std

    def prob0_overlap_machine(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.overlap_circuit(θ_A, θ_B)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits, 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]
        prob0 = np.array([r[0] / sum(r.values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        return mean, std

    def prob0_identity_exact(self, θ):
        circuit = self.identity_circuit(θ)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_identity_noisy(self, θ, p=None, shots=None, repeats=None):
        if p is None:
            p = self.p

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        probs = []
        for _ in range(repeats):
            noise = cirq.depolarize(p=p)
            circuit = self.identity_circuit(θ)

            circuit = AddMeasure(circuit, self.qubits, 'meas')
            result = SimulateCircuitLocalNoisyV(circuit, shots, noise).histogram(key='meas')

            prob0 = result[0] / sum(result.values())
            probs.append(prob0)

        p0_mean = np.mean(probs)
        p0_std = np.std(probs)
        return p0_mean, p0_std

    def prob0_identity_machine(self, θ, shots=None, repeats=None,
                               processor=None, floquet=None):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.identity_circuit(θ)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits, 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]
        prob0 = np.array([r[0] / sum(r.values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        return mean, std

class HalfTrotterGenerator():
    def __init__(self, U, n, qubits=None, Ansatz=StateAnsatzXZ, U2 = None):
        self.n = n
        self.qubits = qubits

        try:
            self.U_layer = self._generate_U(U)
            self.id_layer = None

        except:
            self.U_layer = {q[0]:self._generate_U_qubits(U,q) for q in qubits}

            self.id_layer = {}

            for q in qubits:
                val = self._generate_U_qubits(U2, q)
                val.extend(cirq.inverse(self._generate_U_qubits(U2,q)))

                self.id_layer[q[0]] = val


        self.Ansatz = Ansatz
        self.repeats = 1
        self.plen=None


        # machine run params
        self.processor = 'rainbow'
        self.shots = 10000

    def _generate_U(self, U):
        ops = []
        for i in range(1, self.n*2 + 1, 2):
            q0, q1 = self.qubits[i], self.qubits[i+1]
            ops.extend(two_qubit_matrix_to_operations(q0, q1, U, True,
                                                      clean_operations=True))
        return ops

    def fix_params(self, param_dict, plen):
        self.fixed_params = param_dict
        self.plen = plen

    def fill_params(self, params):
        fparams = np.zeros(self.plen)
        j = 0
        for i in range(self.plen):
            if i in self.fixed_params.keys():
                fparams[i] = self.fixed_params[i]
            else:
                fparams[i] = params[j]
                j += 1
        return fparams

    def _generate_U_qubits(self, U, qubits):
        ops = []
        for i in range(1, self.n*2 + 1, 2):
            q0, q1 = qubits[i], qubits[i+1]
            ops.extend(two_qubit_matrix_to_operations(q0, q1, U, True,
                                                      clean_operations=True))
        return ops


    def set_small_dt_layer(self, I):
        self.id_layer = self._generate_U(I)

    def set_shots(self, shots):
        self.shots = shots

    def overlap_circuit_qubits(self, θ_A, θ_B, qubits):
        circuitA = NSiteCircuit_Ansatz(θ_A, qubits, self.n*2,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, qubits, self.n*2,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer[qubits[0]])
        circuit.append(cirq.inverse(circuitB))

        return circuit

    def overlap_circuit_qubitsp1(self, θ_A, θ_B, qubits):
        circuitA = NSiteCircuit_Ansatz(θ_A, qubits, self.n*2 + 1,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, qubits, self.n*2 + 1,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer[qubits[0]])
        circuit.append(cirq.inverse(circuitB))

        return circuit


    def overlap_circuit(self, θ_A, θ_B):
        circuitA = NSiteCircuit_Ansatz(θ_A, self.qubits, self.n*2,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, self.qubits, self.n*2,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))

        return circuit

    def overlap_circuit_UU(self, θ_A, θ_B, env_no=0):
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env
        site_no = self.n*2 + 1 - env_no

        circuitA = NSiteCircuit_Ansatz_Env(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz_Env(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def overlap_circuit_C1(self, θ_A, θ_B):
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env
        site_no = self.n*2
        env_no = 0
        circuitA = NSiteCircuit_Ansatz_Env(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz_Env(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def overlap_circuit_C1UU(self, θ_A, θ_B):
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env
        site_no = self.n*2
        env_no = 1
        circuitA = NSiteCircuit_Ansatz_Env(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz_Env(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit


    def overlap_circuit_VVprime(self, θ_A, θ_B):
        from qmps.tools import get_env_exact
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env, Gate_to_Unitary
        env_no = 0
        site_no = self.n*2

        UA = Gate_to_Unitary(θ_A, self.Ansatz)
        UB = Gate_to_Unitary(θ_B, self.Ansatz)

        VA = get_env_exact(UA)
        VB = get_env_exact(UB)

        circuitA = [cirq.ops.MatrixGate(VA).on(self.qubits[site_no], self.qubits[site_no+1])]
        circuitA_ = NSiteCircuit_Ansatz_Env(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz, offset=1)
        circuitA.extend(circuitA_)
        circuitB = [cirq.ops.MatrixGate(VB).on(self.qubits[site_no], self.qubits[site_no+1])]
        circuitB_ = NSiteCircuit_Ansatz_Env(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz, offset=1)
        circuitB.extend(circuitB_)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def overlap_circuit_VV(self, θ_A, θ_B):
        from qmps.tools import get_env_exact
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env, Gate_to_Unitary
        env_no = 0
        site_no = self.n*2

        UA = Gate_to_Unitary(θ_A, self.Ansatz)

        VA = get_env_exact(UA)

        circuitA = [cirq.ops.MatrixGate(VA).on(self.qubits[site_no], self.qubits[site_no+1])]
        circuitA_ = NSiteCircuit_Ansatz_Env(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz, offset=1)
        circuitA.extend(circuitA_)
        circuitB = [cirq.ops.MatrixGate(VA).on(self.qubits[site_no], self.qubits[site_no+1])]
        circuitB_ = NSiteCircuit_Ansatz_Env(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz, offset=1)
        circuitB.extend(circuitB_)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def overlap_circuit_UU0(self, θ_A, θ_B, site_no=1, env_no=0):
        from qmpsyc.circuits import NSiteCircuit_Ansatz_Env0

        circuitA = NSiteCircuit_Ansatz_Env0(θ_A, θ_A, self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz_Env0(θ_B, θ_A,  self.qubits, site_no,
                                           env_no, Ansatz=self.Ansatz)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def overlap_circuit_identity(self, θ_A, θ_B):
        '''Generate overlap circuit terminating with identity (i.e. no env)
        '''
        site_no = self.n * 2
        env_no = 0
        return self.overlap_circuit_UU0(θ_A, θ_B, site_no=site_no, env_no=env_no)

    def overlap_circuit_identity(self, θ_A, θ_B):
        '''Generate overlap circuit terminating with identity (i.e. no env)
        '''
        site_no = self.n * 2
        env_no = 0
        return self.overlap_circuit_UU0(θ_A, θ_B, site_no=site_no, env_no=env_no)


    def overlap_circuit_p1(self, θ_A, θ_B):
        circuitA = NSiteCircuit_Ansatz(θ_A, self.qubits, self.n*2 + 1,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, self.qubits, self.n*2 + 1,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))

        return circuit

    def overlap_circuit_p2(self, θ_A, θ_B):
        circuitA = NSiteCircuit_Ansatz(θ_A, self.qubits, self.n*2 + 2,
                                       Ansatz=self.Ansatz)
        circuitB = NSiteCircuit_Ansatz(θ_B, self.qubits, self.n*2 + 2,
                                       Ansatz=self.Ansatz)

        # Construct circuit
        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.U_layer)
        circuit.append(cirq.inverse(circuitB))

        return circuit

    def identity_circuit(self, θ):
        assert self.id_layer is not None, 'Need to set identity circuit'
        circuitA = NSiteCircuit_Ansatz(θ, self.qubits, self.n*2,
                                       Ansatz=self.Ansatz)
        circuitA_inv = cirq.inverse(circuitA)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.id_layer)
        circuit.append(circuitA_inv)

        return circuit

    def identity_circuit_qubits(self, θ, qubits):
        circuitA = NSiteCircuit_Ansatz(θ, qubits, self.n*2,
                                       Ansatz=self.Ansatz)
        circuitA_inv = cirq.inverse(circuitA)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.id_layer[qubits[0]])
        circuit.append(circuitA_inv)

        return circuit

    def identity_circuit_qubitsp1(self, θ, qubits):
        circuitA = NSiteCircuit_Ansatz(θ, qubits, self.n*2 + 1,
                                       Ansatz=self.Ansatz)
        circuitA_inv = cirq.inverse(circuitA)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(self.id_layer[qubits[0]])
        circuit.append(circuitA_inv)

        return circuit

    def prob0_overlap_exact(self, θ_A=None, θ_B=None, circuit=None):
        if circuit==None:
            circuit = self.overlap_circuit_p1(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_overlap_VV_no_post(self, θB, θA):
        circuit = self.overlap_circuit_VV(θA, θB)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_overlap_VVprime_no_post(self, θB, θA):
        circuit = self.overlap_circuit_VVprime(θA, θB)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_overlap_exactp2(self, θ_A, θ_B):
        circuit = self.overlap_circuit_p2(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0


    def prob0_overlap_identity(self, θ_A, θ_B):
        circuit = self.overlap_circuit_identity(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def prob0_overlap_exact_UU(self, θ_A, θ_B, env_no=0):
        circuit = self.overlap_circuit_UU(θ_A, θ_B, env_no=env_no)

        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0

    def probs_overlap_exact(self, θ_A, θ_B):
        circuit = self.overlap_circuit_p1(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)

        probs = np.abs(result.final_state_vector, dtype=float) ** 2
        result_dict = {i: probs[i] for i in range(len(probs))}

        return result_dict

    def probs_overlap_exactp2(self, θ_A, θ_B):
        circuit = self.overlap_circuit_p2(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)

        probs = np.abs(result.final_state_vector, dtype=float) ** 2
        result_dict = {i: probs[i] for i in range(len(probs))}

        return result_dict

    def probs_overlap_exact_UU(self, θ_A, θ_B, env_no=0):
        circuit = self.overlap_circuit_UU(θ_A, θ_B, env_no=env_no)
        result = SimulateCircuitLocalExact(circuit)

        probs = np.abs(result.final_state_vector, dtype=float) ** 2
        result_dict = {i: probs[i] for i in range(len(probs))}

        return result_dict

    def probs_right_env_exact(self, θ_A, θ_B):
        circuit = self.right_env_circuit(θ_A, θ_B)
        result = SimulateCircuitLocalExact(circuit)
        probs = np.abs(result.final_state_vector, dtype=float)**2

        result_dict = {i: probs[i] for i in range(len(probs))}
        return result_dict

    def prob0_right_env_exact(self, θ_A, θ_B):
        results = self.probs_right_env_exact(θ_A, θ_B)

        return results[0] / sum(results.values())

    def prob0_overlap_exact_qubits(self, θ_A, θ_B, qubits):
        circuit = self.overlap_circuit_qubits(θ_A, θ_B, qubits)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0


    def prob0_overlap_machine_qubits(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None, returnall = False):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuits = [self.overlap_circuit_qubits(θ_A, θ_B, qubit) for qubit in self.qubits]

        circuits = [cg.optimizers.optimized_for_sycamore(circuit) for circuit in circuits]

        circuits = [AddMeasure(circuit, self.qubits[i], f'meas_{i}') for i, circuit in enumerate(circuits)]

        combined_circuits = cirq.Circuit()
        for circuit in circuits:
            for moment in circuit:
                combined_circuits.append(moment)

        combined_circuits = cirq.stratified_circuit(combined_circuits, categories = [lambda op: len(op.qubits) == 2])

        result = SimulateGoogleBatched(combined_circuits,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)

        result = [[r.histogram(key=f'meas_{i}') for r in result] for i in range(len(circuits))]
        prob0 = np.array([r[0][0] / sum(r[0].values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        if returnall:
            return  prob0

        else:
            return mean, std

    def prob0_overlap_machine(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.overlap_circuit(θ_A, θ_B)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits, 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]
        prob0 = np.array([r[0] / sum(r.values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        return mean, std

    def overlap_machine_results(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.overlap_circuit_p1(θ_A, θ_B)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits[:-1], 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]

        return result

    def overlap_machine_qubits_results(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None, returnall = False, returnraw = False):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuits = [self.overlap_circuit_qubitsp1(θ_A, θ_B, qubit) for qubit in self.qubits]

        circuits = [cg.optimizers.optimized_for_sycamore(circuit) for circuit in circuits]

        circuits = [AddMeasure(circuit, self.qubits[i][:-1], f'meas_{i}') for i, circuit in enumerate(circuits)]

        combined_circuits = cirq.Circuit()
        for circuit in circuits:
            for moment in circuit:
                combined_circuits.append(moment)

        combined_circuits = cirq.stratified_circuit(combined_circuits, categories = [lambda op: len(op.qubits) == 2])

        result = SimulateGoogleBatched(combined_circuits,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)

        if returnraw:
            return result

        result = [[r.histogram(key=f'meas_{i}') for i in range(len(circuits))] for r in result]

        return result


    def prob0_identity_exact(self, θ):
        circuit = self.identity_circuit(θ)
        result = SimulateCircuitLocalExact(circuit)
        prob0 = np.abs(result.final_state_vector[0], dtype=float) ** 2
        return prob0, 0.0

    def prob0_identity_machine(self, θ, shots=None, repeats=None,
                               processor=None, floquet=None):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.identity_circuit(θ)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits, 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]
        prob0 = np.array([r[0] / sum(r.values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        return mean, std

    def prob0_identity_machine_iswap(self, θ, shots=None, repeats=None,
                               processor=None, floquet=None):
        ''' Run with removed repeated iswaps
        '''
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuit = self.identity_circuit(θ)
        circuit = cg.optimizers.optimized_for_sycamore(circuit)

        if floquet is not None:
            circuit = ApplyFloquet(circuit, floquet)

        circuit = cirq.stratified_circuit(circuit, categories = [lambda op: len(op.qubits) == 2])
        circuit = AddMeasure(circuit, self.qubits, 'meas')

        result = SimulateGoogleBatched(circuit,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)
        result = [r.histogram(key='meas') for r in result]
        prob0 = np.array([r[0] / sum(r.values()) for r in result])
        mean = np.mean(prob0)
        std = np.std(prob0)

        return mean, std

    def prob0_identity_qubits(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None, returnall=False):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuits = [self.identity_circuit_qubits(θ_A, qubit) for qubit in self.qubits]

        circuits = [cg.optimizers.optimized_for_sycamore(circuit) for circuit in circuits]

        circuits = [AddMeasure(circuit, self.qubits[i], f'meas_{i}') for i, circuit in enumerate(circuits)]

        combined_circuits = cirq.Circuit()
        for circuit in circuits:
            for moment in circuit:
                combined_circuits.append(moment)

        combined_circuits = cirq.stratified_circuit(combined_circuits, categories = [lambda op: len(op.qubits) == 2])

        result = SimulateGoogleBatched(combined_circuits,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)

        result = [[r.histogram(key=f'meas_{i}') for r in result] for i in range(len(circuits))]
        prob0 = np.array([r[0][0] / sum(r[0].values()) for r in result])

        if returnall:
            return prob0

        else:
            mean = np.mean(prob0)
            std = np.std(prob0)
            return mean, std

    def results_identity_qubits(self, θ_A, θ_B, shots=None, repeats=None, processor=None, floquet=None, returnall=False, returnraw = False):
        if processor is None:
            processor = self.processor

        if shots is None:
            shots = self.shots

        if repeats is None:
            repeats = self.repeats

        circuits = [self.identity_circuit_qubitsp1(θ_A, qubit) for qubit in self.qubits]

        circuits = [cg.optimizers.optimized_for_sycamore(circuit) for circuit in circuits]

        circuits = [AddMeasure(circuit, self.qubits[i][:-1], f'meas_{i}') for i, circuit in enumerate(circuits)]

        combined_circuits = cirq.Circuit()
        for circuit in circuits:
            for moment in circuit:
                combined_circuits.append(moment)

        combined_circuits = cirq.stratified_circuit(combined_circuits, categories = [lambda op: len(op.qubits) == 2])

        result = SimulateGoogleBatched(combined_circuits,
                                       processor=processor,
                                       BatchNum=repeats,
                                       Reps=shots)

        if returnraw:
            return result.histogtam

        result = [[r.histogram(key=f'meas_{i}') for i in range(len(circuits))] for r in result]
        return result

def remove_repeated_moments(circuit, moment1, moment2):
    circuit_ = []
    skip_next = False
    for mi, mip1 in zip(circuit[:-1], circuit[1:]):
        if mi.operations == moment1 and mip1.operations == moment2:
            skip_next=True
            continue
        elif skip_next:
            skip_next=False
        else:
            circuit_.append(mi)
    if not skip_next:
        circuit_.append(circuit[-1])
    return cirq.Circuit(circuit_)

def remove_repeated_iswaps(circuit, qubits):
    circuit_ = copy(circuit)

    for q0, q1 in zip(qubits[:-1], qubits[1:]):
        m1 = ((cirq.ISWAP**-0.5).on(q0, q1), )
        m2 = ((cirq.ISWAP**0.5).on(q0, q1), )
        circuit_ = remove_repeated_moments(circuit_, m1, m2)

    return circuit_

def calculate_marginals(results, original_len, ignored_indices):
    remaining_indices = [i for i in range(original_len) if i not in ignored_indices]

    new_results_dict = {}
    for i in results.keys():
        i_bin = format(i, 'b').zfill(original_len)
        new_i = [i_bin[i] for i in remaining_indices]
        new_i = ''.join(new_i)
        new_key = int(new_i, 2)

        new_results_dict[new_key] = new_results_dict.get(new_key, 0) + results[i]

    return new_results_dict

def post_select_indices(bin_str, indices):
    '''
    Check if the string matches the post selection criteria
    '''
    for i in indices:
        if bin_str[i] != '0':
            return False
    return True

def calculate_post_selection(results, original_len, post_indices):
    remaining_indices = [i for i in range(original_len) if i not in post_indices]

    new_results_dict = {}

    for i in results.keys():
        i_bin = format(i, 'b').zfill(original_len)
        if post_select_indices(i_bin, post_indices):
            new_i = [i_bin[i] for i in remaining_indices]
            new_i = ''.join(new_i)
            new_key = int(new_i, 2)
            new_results_dict[new_key] = new_results_dict.get(new_key, 0) + results[i]

    return new_results_dict

def process_results_marginal(res, n, qno):
    probs0_n = []
    for i in range(1, n):
        excluded_indices = range(2*i + 1, qno)
        res_n = calculate_marginals(res, qno, excluded_indices)
        p0n = res_n[0] / sum(res_n.values())
        probs0_n.append(p0n)

    probs0_n.append(res[0] / sum(res.values()))

    probs0_n = np.array(probs0_n)

    return probs0_n

def analyse_overlaps_single(results, n):
    qno = 2*n + 1

    results_lst = []
    for res in results:
        res_lst = []
        for i in range(1, n+1):
            excluded_indices = range(2*i + 1, qno)
            res_n = calculate_marginals(res, qno, excluded_indices)
            p0n = res_n[0] / sum(res_n.values())
            res_lst.append(p0n)
        results_lst.append(np.array(res_lst))
    overlaps_lst = [probs[1:]/probs[:-1] for probs in results_lst]
    return np.array(overlaps_lst)

def analyse_probs_single(results, n):
    qno = 2*n + 1

    results_lst = []
    for res in results:
        res_lst = []
        for i in range(1, n+1):
            excluded_indices = range(2*i + 1, qno)
            res_n = calculate_marginals(res, qno, excluded_indices)
            p0n = res_n[0] / sum(res_n.values())
            res_lst.append(p0n)
        results_lst.append(np.array(res_lst))
    return results_lst

def analyse_overlaps_post_select(results, n):
    qno = 2*n + 1

    results_lst = []
    for res in results:
        res_lst = []
        for i in range(1, n+1):
            excluded_indices = range(2*i + 1, qno)
            res_n = calculate_post_selection(res, qno, excluded_indices)
            p0n = res_n[0] / sum(res_n.values())
            res_lst.append(p0n)
        results_lst.append(np.array(res_lst))
    overlaps_lst = [probs[1:]/probs[:-1] for probs in results_lst]

    return np.array(overlaps_lst)

def analyse_probs_post_select(results, n, qno=None):
    if qno is None:
        qno = 2*n + 1

    results_lst = []
    for res in results:
        res_lst = []
        for i in range(1, n+1):
            excluded_indices = range(2*i + 1, qno)
            res_n = calculate_post_selection(res, qno, excluded_indices)
            p0n = res_n[0] / sum(res_n.values())
            res_lst.append(p0n)
        results_lst.append(np.array(res_lst))

    return results_lst

