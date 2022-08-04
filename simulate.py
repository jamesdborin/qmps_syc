import cirq
import numpy as np
import cirq_google as cg
import qsimcirq as qsim
from copy import deepcopy
from floquet import ApplyFloquet
import os

try:
    project_id = os.environ['QGC_KEY']
except:
    print('Using default project_id')
    project_id = 'erudite-mote-326609'
ProjectEngine = cg.get_engine(project_id)

def SimulateCircuitLocalNoiseless(Circuit, Reps):
    # Simulate circuit locally with sampling but no noise

    sim = cirq.Simulator()

    results = sim.run(Circuit, repetitions = Reps)
    return results


def SimulateCircuitLocalNoisy(Circuit, Reps, Noise):
    # Simulate Circuits Locally with sampling and a noise model specified by Noise
    # e.g. Noise = cirq.depolarize(p = 0.01)
    # This is much slower so we are going to use qsimcirq

    Circuit = cg.optimizers.optimized_for_sycamore(Circuit)
    sim = qsim.QSimSimulator({'t':6})
    results = sim.run(Circuit.with_noise(Noise), repetitions = Reps)

    return results

def SimulateCircuitLocalNoisyV(Circuit, Reps, Noise):
    # Simulate Circuits Locally with sampling and a noise model specified by Noise
    # e.g. Noise = cirq.depolarize(p = 0.01)
    # This is much slower so we are going to use qsimcirq

    Circuit = cg.optimizers.optimized_for_sycamore(Circuit)
    sim = qsim.QSimSimulator({'t':6})
    results = sim.run(Circuit.with_noise(Noise), repetitions = Reps)

    return results


def SimulateCircuitLocalClassicalReadoutError(Circuit, MeasureQubits, Reps, P):
    # Simulate circuits locally with sampling - using a noise model that simulates
    #   classical readout error:

    sim = qsim.QSimSimulator({'t':6})

    noisyCircuit = deepcopy(Circuit)
    noisyCircuit.insert(-1, cirq.bit_flip(p=P).on_each(MeasureQubits))
    #print(noisyCircuit.to_text_diagram(transpose = True))
    results = sim.run(noisyCircuit, repetitions = Reps)

    return results



def SimulateCircuitLocalExact(Circuit, Reps = None, dtype=np.complex64):
    # Simulate circuit locally without sampling

    sim = cirq.Simulator(dtype=dtype)

    results = sim.simulate(Circuit)
    return results


def FloquetCalibration(circuit, device_sampler, **kwargs):
    (calibrated_circuit, characterizations) = cg.run_zeta_chi_gamma_compensation_for_moments(
        circuit,
        device_sampler.sampler,
        **kwargs
    )

    return calibrated_circuit.circuit

def SimulateCircuitGoogle(Circuit, Reps, Floquet = False, Characterizations = None, engine = ProjectEngine, processor='weber'):
    #print(project_id)
    # Simulate circuit on google hardware


    #Circuit = cg.optimizers.optimized_for_sycamore(Circuit)

    if Floquet:
        assert Characterizations is not None
        Circuit = ApplyFloquet(Circuit, Characterizations)


    results = engine.run(
        Circuit,
        processor_ids = [processor],
        gate_set=cg.SQRT_ISWAP_GATESET,
        repetitions = Reps
        )

    return results


def SimulateGooglePreBatched(BatchedCircuits, Reps, Floquet, Characterizations, CharacterizationKeys, engine=ProjectEngine, processor='weber'):
    # Simulate a set of circuits in a single batch which share measure qubits

    #engine = cg.get_engine()
    # BatchedCircuits = [cg.optimizers.optimized_for_sycamore(c) for c in BatchedCircuits]

    if Floquet:
        assert Characterizations is not None
        BatchedCircuits = [ApplyFloquet(c, Characterizations[key]) for c,key in zip(BatchedCircuits, CharacterizationKeys)]


    results = engine.run_batch(
        BatchedCircuits,
        processor_ids = [processor],
        gate_set=cg.SQRT_ISWAP_GATESET,
        repetitions = Reps
        )

    return results


def SimulateGoogleBatched(Circuit, Reps, BatchNum, Floquet = False, Characterizations = None, engine=ProjectEngine, processor='weber'):
    # Simulate a single circuit BatchNum times

    # engine = cg.get_engine()
    #Circuit = cg.optimizers.optimized_for_sycamore(Circuit)
    if Floquet:
        assert Characterizations is not None
        Circuit = ApplyFloquet(Circuit, Characterizations)

    batched_circuits = [Circuit for _ in range(BatchNum)]

    results = engine.run_batch(
        programs = batched_circuits,
        processor_ids = [processor],
        gate_set=cg.SQRT_ISWAP_GATESET,
        repetitions = Reps
        )

    return results
