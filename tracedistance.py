import numpy as np
import cirq_google as cg
from qmpsyc.circuits import (
    SSSt, JustEnv, 
    SPCircuit, SECircuit, EPCircuit, 
    AddMeasure,
    D4JustEnv, D4SingleSiteCircuit
    )

from qmpsyc.postpro import (
    SampledTrace, ExactTrace, SampledTraceCM
    )

from qmpsyc.simulate import (
    SimulateCircuitLocalExact,
    SimulateCircuitLocalNoiseless,
    SimulateCircuitGoogle,
    SimulateGoogleBatched,
    SimulateCircuitLocalNoisy,
    SimulateGooglePreBatched,
    SimulateCircuitLocalClassicalReadoutError
)

from qmpsyc.nu_op_calibration import nuop


def TraceDistanceAnalytic(Th, Psi, Q):
    # Calcualte trace distance by comparing the reduced density matrices directly of a state and state+env pair
    SingleSiteState = SSSt(Th, Psi, Q[:3])
    SingleEnv = JustEnv(Th, Q[:2])

    SiteRes = SimulateCircuitLocalExact(SingleSiteState)
    EnvRes = SimulateCircuitLocalExact(SingleEnv)

    rho = SiteRes.density_matrix_of([Q[0]])
    sig = EnvRes.density_matrix_of([Q[0]])

    return np.linalg.norm( rho - sig )**2


def TraceDistanceSampled(Th, Psi, Q, Reps):
    # Calculate trace distance using three circuits with smapling 
    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SP = AddMeasure(SP, Q[2:4], 'SP')
    SE = AddMeasure(SE, Q[1:3], 'SE')
    EP = AddMeasure(EP, Q[1:3], 'EP')

    SPRes = SimulateCircuitLocalNoiseless(SP, Reps).histogram(key = 'SP')
    SERes = SimulateCircuitLocalNoiseless(SE, Reps).histogram(key = 'SE')
    EPRes = SimulateCircuitLocalNoiseless(EP, Reps).histogram(key = 'EP')

    SPTrace = SampledTrace(SPRes)
    SETrace = SampledTrace(SERes)
    EPTrace = SampledTrace(EPRes)

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceUnSampled(Th, Psi, Q):
    # Calculate the trace distance using three circuits with density matrix simulation
    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SPRes = SimulateCircuitLocalExact(SP)
    SERes = SimulateCircuitLocalExact(SE)
    EPRes = SimulateCircuitLocalExact(EP)

    SPTrace = ExactTrace(SPRes, Q[2:4]).real
    SETrace = ExactTrace(SERes, Q[1:3]).real
    EPTrace = ExactTrace(EPRes, Q[1:3]).real

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceGoogle(Th, Psi, QSP, QSE, QEP, Reps, Floquet=False, Characterizations=None, processor = 'weber'):
    if Characterizations is None:
        Characterizations = {'SP':None, 'EP':None,'SE':None}
    
    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)


    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    SPRes = SimulateCircuitGoogle(SP, Reps, Floquet, Characterizations['SP'], processor=processor).histogram(key = 'SP')
    SERes = SimulateCircuitGoogle(SE, Reps, Floquet, Characterizations['SE'], processor=processor).histogram(key = 'SE')
    EPRes = SimulateCircuitGoogle(EP, Reps, Floquet, Characterizations['EP'], processor=processor).histogram(key = 'EP')

    SPTrace = SampledTrace(SPRes)
    SETrace = SampledTrace(SERes)
    EPTrace = SampledTrace(EPRes)

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceGoogleBatched(Th, Psi, QSP, QSE, QEP, Reps, BatchNum, Floquet = False, Characterizations = None, processor = 'weber'):
    if Characterizations is None:
        Characterizations = {'SP':None, 'EP':None,'SE':None}

    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )
    
    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    # Get the raw data i.e. not counter data
    SPResRaw = SimulateGoogleBatched(SP, Reps, BatchNum, Floquet, Characterizations['SP'],processor=processor)
    SEResRaw = SimulateGoogleBatched(SE, Reps, BatchNum, Floquet, Characterizations['SE'],processor=processor)
    EPResRaw = SimulateGoogleBatched(EP, Reps, BatchNum, Floquet, Characterizations['EP'],processor=processor)

    SPRes = [res.histogram(key = 'SP') for res in SPResRaw]
    SERes = [res.histogram(key = 'SE') for res in SEResRaw]
    EPRes = [res.histogram(key = 'EP') for res in EPResRaw]


    SPTrace = np.array([SampledTrace(res) for res in SPRes])
    SETrace = np.array([SampledTrace(res) for res in SERes])
    EPTrace = np.array([SampledTrace(res) for res in EPRes])

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceNoisy(Th, Psi, Q, Reps, Noise):
    # Calculate trace distance using three circuits with smapling 
    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, Q[2:4], 'SP')
    SE = AddMeasure(SE, Q[1:3], 'SE')
    EP = AddMeasure(EP, Q[1:3], 'EP')

    SPRes = SimulateCircuitLocalNoisy(SP, Reps, Noise).histogram(key = 'SP')
    SERes = SimulateCircuitLocalNoisy(SE, Reps, Noise).histogram(key = 'SE')
    EPRes = SimulateCircuitLocalNoisy(EP, Reps, Noise).histogram(key = 'EP')

    SPTrace = SampledTrace(SPRes)
    SETrace = SampledTrace(SERes)
    EPTrace = SampledTrace(EPRes)

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceSampledCorrected(Th, Psi, Q, Reps, invCM):
    """Calculate trace distance using three circuits with sampling
    Also use an inverted confusion matrix to improve the results
    """

    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, Q[2:4], 'SP')
    SE = AddMeasure(SE, Q[1:3], 'SE')
    EP = AddMeasure(EP, Q[1:3], 'EP')

    SPRes = SimulateCircuitLocalNoiseless(SP, Reps).histogram(key = 'SP')
    SERes = SimulateCircuitLocalNoiseless(SE, Reps).histogram(key = 'SE')
    EPRes = SimulateCircuitLocalNoiseless(EP, Reps).histogram(key = 'EP')

    SPTrace = SampledTraceCM(SPRes, invCM)
    SETrace = SampledTraceCM(SERes, invCM)
    EPTrace = SampledTraceCM(EPRes, invCM)

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceGoogleCorrected(Th, Psi, QSP, QSE, QEP, Reps, invCM, Floquet=False, Characterizations=None, processor = 'weber'):
    """Use google hardware to estimate the trace distance.
    Answer is improved by providing 3 inverted confusion matrices for each of the measured qubit sets.
    
    Provide characterisations for the three circuits in a dictionary to apply cheap floquet calibration"""

    if isinstance( invCM, np.ndarray ):
        invCM = [invCM, invCM, invCM]
    if Characterizations is None:
        Characterizations = {'SP':None, 'SE':None,'EP':None}

    
    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    SPRes = SimulateCircuitGoogle(SP, Reps, Floquet, Characterizations['SP'],processor=processor).histogram(key = 'SP')
    SERes = SimulateCircuitGoogle(SE, Reps, Floquet, Characterizations['SE'],processor=processor).histogram(key = 'SE')
    EPRes = SimulateCircuitGoogle(EP, Reps, Floquet, Characterizations['EP'],processor=processor).histogram(key = 'EP')

    SPTrace = SampledTraceCM(SPRes, invCM[0])
    SETrace = SampledTraceCM(SERes, invCM[1])
    EPTrace = SampledTraceCM(EPRes, invCM[2])

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceGoogleCorrectedBatched(Th, Psi, QSP, QSE, QEP, Reps, invCM, BatchNum, Floquet=False, Characterizations=None, processor = 'weber'):
    """Simualate the trace distance of google hardware, batching the result to gain error bars for a single measurement"""
    if isinstance( invCM, np.ndarray ):
        invCM = [invCM, invCM, invCM]

    if Characterizations is None:
        Characterizations = {'SP':None, 'SE':None,'EP':None}


    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    SPResRaw = SimulateGoogleBatched(SP, Reps, BatchNum, Floquet, Characterizations['SP'],processor=processor)
    SEResRaw = SimulateGoogleBatched(SE, Reps, BatchNum, Floquet, Characterizations['SE'],processor=processor)
    EPResRaw = SimulateGoogleBatched(EP, Reps, BatchNum, Floquet, Characterizations['EP'],processor=processor)

    SPRes = [res.histogram(key = 'SP') for res in SPResRaw]
    SERes = [res.histogram(key = 'SE') for res in SEResRaw]
    EPRes = [res.histogram(key = 'EP') for res in EPResRaw]

    SPTrace = np.array([SampledTraceCM(res, invCM[0]) for res in SPRes])
    SETrace = np.array([SampledTraceCM(res, invCM[1]) for res in SERes])
    EPTrace = np.array([SampledTraceCM(res, invCM[2]) for res in EPRes])

    return SPTrace + EPTrace - 2*SETrace

def TraceDistanceGoogleCorrectedBatchedSeparate(Th, Psi, QSP, QSE, QEP, Reps, invCM, BatchNum, Floquet=False, Characterizations=None, processor = 'weber'):
    """Simualate the trace distance of google hardware, batching the result to gain error bars for a single measurement"""
    if isinstance( invCM, np.ndarray ):
        invCM = [invCM, invCM, invCM]

    if Characterizations is None:
        Characterizations = {'SP':None, 'SE':None,'EP':None}


    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    SPResRaw = SimulateGoogleBatched(SP, Reps, BatchNum, Floquet, Characterizations['SP'],processor=processor)
    SEResRaw = SimulateGoogleBatched(SE, Reps, BatchNum, Floquet, Characterizations['SE'],processor=processor)
    EPResRaw = SimulateGoogleBatched(EP, Reps, BatchNum, Floquet, Characterizations['EP'],processor=processor)

    SPRes = [res.histogram(key = 'SP') for res in SPResRaw]
    SERes = [res.histogram(key = 'SE') for res in SEResRaw]
    EPRes = [res.histogram(key = 'EP') for res in EPResRaw]

    SPTrace = np.array([SampledTraceCM(res, invCM[0]) for res in SPRes])
    SETrace = np.array([SampledTraceCM(res, invCM[1]) for res in SERes])
    EPTrace = np.array([SampledTraceCM(res, invCM[2]) for res in EPRes])

    return SPTrace, EPTrace, SETrace


def TraceDistanceGoogleCorrectedTripleBatched(Th, Psi, QSP, QSE, QEP, Reps, invCM, Floquet=False, Characterizations=None, processor = 'weber'):
    """For optimization we want to batch all three trace distance circuits together, so we use this triple batched function."""
    if isinstance( invCM, np.ndarray ):
        invCM = [invCM, invCM, invCM]

    SP = SPCircuit( Th, Psi, QSP )
    SE = SECircuit( Th, Psi, QSE )
    EP = EPCircuit( Th, QEP )

    SP = cg.optimizers.optimized_for_sycamore(SP)
    SE = cg.optimizers.optimized_for_sycamore(SE)
    EP = cg.optimizers.optimized_for_sycamore(EP)

    SP = AddMeasure(SP, QSP[2:4], 'SP')
    SE = AddMeasure(SE, QSE[1:3], 'SE')
    EP = AddMeasure(EP, QEP[1:3], 'EP')

    batched_circuits = [SP,SE,EP]

    TraceDistanceRes = SimulateGooglePreBatched(batched_circuits, Reps, Floquet, Characterizations, ['SP','SE','EP'], processor=processor )

    SPTrace = SampledTraceCM(TraceDistanceRes[0].histogram(key='SP'), invCM[0])
    SETrace = SampledTraceCM(TraceDistanceRes[1].histogram(key='SE'), invCM[1])
    EPTrace = SampledTraceCM(TraceDistanceRes[2].histogram(key='EP'), invCM[2])

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceClassicalError(Th, Psi, Q, Reps, P):
    """Measuer the trace distance, simulating classical readout errors""" 
    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SP = AddMeasure(SP, Q[2:4], 'SP')
    SE = AddMeasure(SE, Q[1:3], 'SE')
    EP = AddMeasure(EP, Q[1:3], 'EP')

    SPRes = SimulateCircuitLocalClassicalReadoutError(SP, MeasureQubits = Q[2:4],Reps=Reps, P=P).histogram(key = 'SP')
    SERes = SimulateCircuitLocalClassicalReadoutError(SE, MeasureQubits = Q[1:3],Reps=Reps, P=P).histogram(key = 'SE')
    EPRes = SimulateCircuitLocalClassicalReadoutError(EP, MeasureQubits = Q[1:3], Reps=Reps, P=P).histogram(key = 'EP')

    SPTrace = SampledTrace(SPRes)
    SETrace = SampledTrace(SERes)
    EPTrace = SampledTrace(EPRes)

    return SPTrace + EPTrace - 2*SETrace


def TraceDistanceClassicalErrorCorrected(Th, Psi, Q, Reps, P, invCM):
    """Simulate the trace distance with classical readout error - which has been corrected by a confusion matrix""" 
    SP = SPCircuit(Th, Psi, Q[:6]) # State Purity 
    SE = SECircuit(Th, Psi, Q[:5]) # State Environment
    EP = EPCircuit(Th, Q[:4])      # Environment Purity

    SP = AddMeasure(SP, Q[2:4], 'SP')
    SE = AddMeasure(SE, Q[1:3], 'SE')
    EP = AddMeasure(EP, Q[1:3], 'EP')

    SPRes = SimulateCircuitLocalClassicalReadoutError(SP, MeasureQubits = Q[2:4],Reps = Reps, P=P).histogram(key = 'SP')
    SERes = SimulateCircuitLocalClassicalReadoutError(SE, MeasureQubits = Q[1:3],Reps=Reps, P=P).histogram(key = 'SE')
    EPRes = SimulateCircuitLocalClassicalReadoutError(EP, MeasureQubits = Q[1:3], Reps=Reps, P=P).histogram(key = 'EP')

    SPTrace = SampledTraceCM(SPRes, invCM)
    SETrace = SampledTraceCM(SERes, invCM)
    EPTrace = SampledTraceCM(EPRes, invCM)

    return SPTrace + EPTrace - 2*SETrace


def D4TraceDistanceAnalytic(state_params, env_params, Q):
    """Calcualte trace distance exactly within this ansatz class"""

    single_site_state = D4SingleSiteCircuit( state_params, env_params, Q[:5] )
    single_env = D4JustEnv(env_params, Q[:4])

    site_results = SimulateCircuitLocalExact(single_site_state)
    env_results = SimulateCircuitLocalExact(single_env)

    rho = site_results.density_matrix_of([Q[1:3]])
    sig = env_results.density_matrix_of([Q[1:3]])

    return np.linalg.norm(rho - sig)**2
