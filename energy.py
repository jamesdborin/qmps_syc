import numpy as np
import cirq_google as cg

from circuits import (
    StCircuit, AddHamTerm, AddMeasure, 
    ZZMeasure, IXMeasure, XIMeasure
)

from postpro import (
    ExactEnergy, SampledEnergy, SampleEnergyCorrected
)

from simulate import (
    SimulateCircuitLocalExact,
    SimulateCircuitLocalNoiseless,
    SimulateCircuitLocalNoisy,
    SimulateCircuitGoogle,
    SimulateGoogleBatched,
    SimulateCircuitLocalClassicalReadoutError,
    SimulateGooglePreBatched
)

def EnergyAnalytic( Th, Psi, Q, H ):
    # Calculate the energy by getting the reduced density matrices over two sites

    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    Res = SimulateCircuitLocalExact(TwoSiteState)

    Rho = Res.density_matrix_of(Q[1:3])

    return np.trace( Rho @ H).real


def EnergyUnsampled( Th, Psi, Q, H ):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    
    Res = SimulateCircuitLocalExact(TwoSiteState)

    E = ExactEnergy(Res, Q[1]).real

    return E


def EnergySampled(Th, Psi, Q, H, Reps):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')
    
    Res = SimulateCircuitLocalNoiseless(TwoSiteState, Reps = Reps).histogram(key = 'E')

    E = SampledEnergy(Res)

    return E


def EnergyNoisy(Th, Psi, Q, H, Reps, Noise):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')
    
    Res = SimulateCircuitLocalNoisy(TwoSiteState, Reps = Reps, Noise = Noise).histogram(key = 'E')

    E = SampledEnergy(Res)
    
    return E


def EnergyGoogle(Th, Psi, Q, H, Reps, Floquet = False, Characterizations = None, processor = 'weber'):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')

    Res = SimulateCircuitGoogle(TwoSiteState, Reps, Floquet, Characterizations, processor=processor).histogram(key = 'E')

    E = SampledEnergy(Res)

    return E

def EnergyGoogleBatched(Th, Psi, Q, H, Reps, BatchNum, Floquet = False, Characterizations = None, processor = 'weber'):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')

    ResRaw = SimulateGoogleBatched(TwoSiteState, Reps, BatchNum, Floquet, Characterizations, processor=processor)

    Res = [res.histogram(key = 'E') for res in ResRaw]

    E = np.array([SampledEnergy(res) for res in Res])

    return E


def EnergySampledCorrected(Th, Psi, Q, H, Reps, invCM):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')
    
    Res = SimulateCircuitLocalNoiseless(TwoSiteState, Reps = Reps).histogram(key = 'E')

    E = SampleEnergyCorrected(Res, invCM)

    return E


def EnergyClassicalErrorCorrected(Th, Psi, Q, H, Reps, P , invCM):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')
    
    Res = SimulateCircuitLocalClassicalReadoutError(TwoSiteState, Reps = Reps, P = P, MeasureQubits=Q[1:3]).histogram(key = 'E')

    E = SampleEnergyCorrected(Res, invCM)
    
    return E


def EnergyClassicalErrorUncorrected(Th, Psi, Q, H, Reps, P):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')
    
    Res = SimulateCircuitLocalClassicalReadoutError(TwoSiteState, Reps = Reps, P = P, MeasureQubits=Q[1:3]).histogram(key = 'E')

    E = SampledEnergy(Res)
    
    return E


def EnergyGoogleCorrected(Th, Psi, Q, H, Reps, invCM, Floquet=False, DeviceSampler=None, processor = 'weber'):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')

    Res = SimulateCircuitGoogle(TwoSiteState, Reps, Floquet, DeviceSampler, processor=processor).histogram(key = 'E')

    E = SampleEnergyCorrected(Res, invCM)

    return E


def EnergyGoogleCorrectedBatched(Th, Psi, Q, H, Reps, invCM, BatchNum, Floquet=False, Characterizations=None, processor = 'weber'):
    TwoSiteState = StCircuit(Th, Psi, Q[:4])
    TwoSiteState = AddHamTerm(TwoSiteState, H, Q[1:3])
    TwoSiteState = cg.optimizers.optimized_for_sycamore(TwoSiteState)
    TwoSiteState = AddMeasure(TwoSiteState, [Q[1]], 'E')

    ResRaw = SimulateGoogleBatched(TwoSiteState, Reps, BatchNum, Floquet, Characterizations, processor=processor)
    
    Res = [res.histogram(key = 'E') for res in ResRaw]

    E = np.array([SampleEnergyCorrected(res, invCM) for res in Res])

    return E


def EnergyGoogleCorrectedTripleBatched(Th, Psi, Q, BatchedH, MeasureQubits, Reps, invCM, Floquet = False, Characterizations = None, CharacterizationKeys=None, processor = 'weber'):
  # Given 3 energy measures we construct three circuits and run them in batch
    BaseCircuits = [StCircuit(Th, Psi, Q) for _ in range(len(BatchedH))]

    batched_circuits = []
    for i,(H,MQ) in enumerate(zip(BatchedH, MeasureQubits)):
        e_circuit = AddHamTerm( BaseCircuits[i], H, Q[1:3])
        e_circuit = cg.optimizers.optimized_for_sycamore(e_circuit)
        e_circuit = AddMeasure( e_circuit, [MQ], 'E' )
        batched_circuits.append(e_circuit)

    Res = SimulateGooglePreBatched( batched_circuits, Reps, Floquet, Characterizations, CharacterizationKeys, processor=processor )

    Es = [SampleEnergyCorrected( res.histogram(key='E'), inv ) for res, inv in zip(Res, invCM)]
    
    return Es


def TFIMEnergyAnalytic(Th, Psi, Q, J, g):
    # Calculate TFIM energy by constructing H = J*zz + g*(XI+IX)/2 and calculating exact energy of this 
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])
    I = np.eye(2)

    ZZ = np.kron(Z,Z)
    IX = np.kron(I,X)
    XI = np.kron(X,I)

    TFIM = J*ZZ + g * (XI + IX) / 2
    E = EnergyAnalytic(Th, Psi, Q, TFIM)

    return E


def TFIMEnergySampled(Th, Psi, Q, Reps, J, g ):
    # Calculate TFIM energy by sampling from a quanutm state
    EZZ = EnergySampled(Th, Psi, Q, ZZMeasure, Reps)
    EXI = EnergySampled(Th, Psi, Q, XIMeasure, Reps)
    EIX = EnergySampled(Th, Psi, Q, IXMeasure, Reps)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyUnSampled(Th, Psi, Q, J, g):
    #  Calculate TFIM energy by looking at density matrix of the measured qubit 
    EZZ = EnergyUnsampled(Th, Psi, Q, ZZMeasure)
    EXI = EnergyUnsampled(Th, Psi, Q, XIMeasure)
    EIX = EnergyUnsampled(Th, Psi, Q, IXMeasure)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyGoogle(Th, Psi, Q, J, g, Reps, Floquet = False, Characterizations = None,processor = 'weber'):
    # Calculate TFIM using google hardware 
    if Characterizations is None:
        Characterizations = {'ZZ':None, 'XI':None,'IX':None}

    EZZ = EnergyGoogle(Th, Psi, Q, ZZMeasure, Reps, Floquet, Characterizations['ZZ'], processor=processor)
    EXI = EnergyGoogle(Th, Psi, Q, XIMeasure, Reps, Floquet, Characterizations['XI'], processor=processor)
    EIX = EnergyGoogle(Th, Psi, Q, IXMeasure, Reps, Floquet, Characterizations['IX'], processor=processor)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyGoogleBatched(Th, Psi, Q, J, g, Reps, BatchNum, Floquet = False, Characterizations = None, processor = 'weber'):
    # Calculate TFIM using google hardware 
    if Characterizations is None:
        Characterizations = {'ZZ':None, 'XI':None,'IX':None}

    EXI = EnergyGoogleBatched(Th, Psi, Q, XIMeasure, Reps, BatchNum, Floquet, Characterizations['XI'], processor=processor)
    EIX = EnergyGoogleBatched(Th, Psi, Q, IXMeasure, Reps, BatchNum, Floquet, Characterizations['IX'], processor=processor)
    EZZ = EnergyGoogleBatched(Th, Psi, Q, ZZMeasure, Reps, BatchNum, Floquet, Characterizations['ZZ'], processor=processor)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyNoisy(Th, Psi, Q, J, g, Reps, Noise):
    # Calculate TFIM using noisy simulators
    EZZ = EnergyNoisy(Th, Psi, Q, ZZMeasure, Reps, Noise)
    EXI = EnergyNoisy(Th, Psi, Q, XIMeasure, Reps, Noise)
    EIX = EnergyNoisy(Th, Psi, Q, IXMeasure, Reps, Noise)

    return J * EZZ + g * (EXI + EIX) / 2



def TFIMEnergyGoogleCorrectedTripleBatched(Th, Psi, Q, Reps, J, g, invCM, Floquet = False, Characterizations = None, processor = 'weber'):
    BatchedHs = [ZZMeasure, XIMeasure, IXMeasure]

    EZZ, EXI, EIX = EnergyGoogleCorrectedTripleBatched( Th, Psi, Q, BatchedHs, Reps, invCM, Floquet, Characterizations, ['ZZ','XI','IX'], processor=processor )

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergySampledCorrected(Th, Psi, Q, Reps, J, g, invCM ):
    # Calculate TFIM energy by sampling from a quanutm state
    EZZ = EnergySampledCorrected(Th, Psi, Q, ZZMeasure, Reps, invCM)
    EXI = EnergySampledCorrected(Th, Psi, Q, XIMeasure, Reps, invCM)
    EIX = EnergySampledCorrected(Th, Psi, Q, IXMeasure, Reps, invCM)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyGoogleCorrected(Th, Psi, Q, J, g, Reps, invCM, Floquet=False, Characterizations=None, processor = 'weber'):
    # Calculate TFIM using google hardware   
    if Characterizations is None:
        Characterizations = {'ZZ':None, 'XI':None,'IX':None}

    EZZ = EnergyGoogleCorrected(Th, Psi, Q, ZZMeasure, Reps, invCM, Floquet, Characterizations['ZZ'], processor=processor)
    EXI = EnergyGoogleCorrected(Th, Psi, Q, XIMeasure, Reps, invCM, Floquet, Characterizations['XI'], processor=processor)
    EIX = EnergyGoogleCorrected(Th, Psi, Q, IXMeasure, Reps, invCM, Floquet, Characterizations['IX'],processor=processor)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyGoogleCorrectedBatched(Th, Psi, Q, J, g, Reps, BatchNum, invCM, Floquet=False, Characterizations=None, processor = 'weber'):
    # Calculate TFIM using google hardware 
    if Characterizations is None:
        Characterizations = {'ZZ':None, 'XI':None,'IX':None}

    EZZ = EnergyGoogleCorrectedBatched(Th, Psi, Q, ZZMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['ZZ'], processor=processor)
    EXI = EnergyGoogleCorrectedBatched(Th, Psi, Q, XIMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['XI'], processor=processor)
    EIX = EnergyGoogleCorrectedBatched(Th, Psi, Q, IXMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['IX'], processor=processor)

    return J * EZZ + g * (EXI + EIX) / 2

def TFIMEnergyGoogleCorrectedBatchedSeparate(Th, Psi, Q, J, g, Reps, BatchNum, invCM, Floquet=False, Characterizations=None, processor = 'weber'):
    # Calculate TFIM using google hardware 
    if Characterizations is None:
        Characterizations = {'ZZ':None, 'XI':None,'IX':None}

    EZZ = EnergyGoogleCorrectedBatched(Th, Psi, Q, ZZMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['ZZ'], processor=processor)
    EXI = EnergyGoogleCorrectedBatched(Th, Psi, Q, XIMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['XI'], processor=processor)
    EIX = EnergyGoogleCorrectedBatched(Th, Psi, Q, IXMeasure, Reps, invCM, BatchNum, Floquet, Characterizations['IX'], processor=processor)

    return EZZ, EXI, EIX


def TFIMEnergyClassicalErrorCorrected(Th, Psi, Q, J, g, Reps, P, invCM):
    # Calculate TFIM using noisy simulators
    EZZ = EnergyClassicalErrorCorrected(Th, Psi, Q, ZZMeasure, Reps, P, invCM)
    EXI = EnergyClassicalErrorCorrected(Th, Psi, Q, XIMeasure, Reps, P, invCM)
    EIX = EnergyClassicalErrorCorrected(Th, Psi, Q, IXMeasure, Reps, P, invCM)

    return J * EZZ + g * (EXI + EIX) / 2


def TFIMEnergyClassicalErrorUnCorrected(Th, Psi, Q, J, g, Reps, P):
    # Calculate TFIM using noisy simulators
    EZZ = EnergyClassicalErrorUncorrected(Th, Psi, Q, ZZMeasure, Reps, P)
    EXI = EnergyClassicalErrorUncorrected(Th, Psi, Q, XIMeasure, Reps, P)
    EIX = EnergyClassicalErrorUncorrected(Th, Psi, Q, IXMeasure, Reps, P)

    return J * EZZ + g * (EXI + EIX) / 2

