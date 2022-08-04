from matplotlib.pyplot import get
import numpy as np
import cirq_google as cg
from circuits import (
    StCircuit,
    AddHamTerm,
    AddMeasure,
    ZZMeasure,
    XIMeasure,
    IXMeasure,
    SPCircuit,
    SECircuit,
    EPCircuit
)

def PrepareCalibration(circuit):
    # Figure out the calibrations needed to be made on the device
    (characterized_circuit, characterization_requests) = cg.prepare_characterization_for_moments(
        circuit,
        options=cg.FloquetPhasedFSimCalibrationOptions(
            characterize_theta=False, 
            characterize_zeta=True,
            characterize_chi=False,  
            characterize_gamma=True,
            characterize_phi=False   
        )
    )

    return characterized_circuit, characterization_requests


def RunCalibration( characterization_requests, device_sampler ):
    # use the google deviec to apply the relevant calibrations

    characterizations = cg.run_calibrations(
        characterization_requests,
        device_sampler.sampler,
        max_layers_per_request=1,
    )

    return characterizations


def MakeCompensations( characterized_circuit, characterizations ):
    calibrated_circuit = cg.make_zeta_chi_gamma_compensation_for_moments(
        characterized_circuit,
        characterizations
    )

    return calibrated_circuit.circuit


def GetCharacterizations(circuit, sampler):
    circuit = cg.optimized_for_sycamore(circuit)
    characterized_circuit, characterization_requests = PrepareCalibration(circuit)
    characterizations = RunCalibration( characterization_requests, sampler )

    return characterizations


def ApplyFloquet(circuit, characterizations):
    # apply floquet calibration given that we have already run the calibrations on the chip

    characterized_circuit, _ = PrepareCalibration( circuit )
    compensated_circuit = MakeCompensations( characterized_circuit, characterizations )

    return compensated_circuit


def FloquetCalibration(circuit, device_sampler, **kwargs):
    (calibrated_circuit, characterizations) = cg.run_zeta_chi_gamma_compensation_for_moments(
        circuit,
        device_sampler.sampler,
        **kwargs
    )

    return calibrated_circuit.circuit


def get_all_calibrations(EQ, QSP, QSE, QEP, device_sampler):
    RandomAngles = np.random.rand(8)
    Th, Psi = RandomAngles[:4], RandomAngles[4:]
    EnergyBaseCircuits = [StCircuit( Th, Psi, EQ ) for _ in range(3)]

    ZZCircuitCal = AddHamTerm(EnergyBaseCircuits[0], ZZMeasure, EQ[1:3])
    ZZCircuitCal = AddMeasure(ZZCircuitCal, [EQ[1]], 'E')

    XICircuitCal = AddHamTerm(EnergyBaseCircuits[1], XIMeasure, EQ[1:3])
    XICircuitCal = AddMeasure(ZZCircuitCal, [EQ[1]], 'E')

    IXCircuitCal = AddHamTerm(EnergyBaseCircuits[2], IXMeasure, EQ[1:3])
    IXCircuitCal = AddMeasure(IXCircuitCal, [EQ[1]], 'E')

    # Trace Distance Circuits

    SPCal = SPCircuit( Th, Psi, QSP )
    SPCal = AddMeasure( SPCal, QSP[2:4], 'SP' )

    SECal = SECircuit( Th, Psi, QSE )
    SECal = AddMeasure( SECal, QSE[1:3], 'SE' )

    EPCal = EPCircuit( Th, QEP )
    EPCal = AddMeasure( EPCal, QEP[1:3], 'EP' )

    # Get the calibrations

    Characterizations = {}

    Characterizations['ZZ'] = GetCharacterizations( ZZCircuitCal, device_sampler )
    Characterizations['XI'] = GetCharacterizations( XICircuitCal, device_sampler )
    Characterizations['IX'] = GetCharacterizations( IXCircuitCal, device_sampler )
    Characterizations['SP'] = GetCharacterizations( SPCal, device_sampler )
    Characterizations['SE'] = GetCharacterizations( SECal, device_sampler )
    Characterizations['EP'] = GetCharacterizations( EPCal, device_sampler )


    return Characterizations


