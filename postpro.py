import numpy as np

def SampledTrace(results):
    # Calculate the trace from sampled data
    
    PFail = results[3] / sum(results.values())
    trace = 1 - 2*PFail

    return trace


def SampledEnergy(results):
    # Calculate Energy from sampled data, assuming everything is in the Z Basis

    num0s = results[0]
    num1s = results[1]

    E = (num0s - num1s) / (num0s + num1s)
    
    return E

def ExactTrace(results, Qs):
    # Calculate trace using density matrix simulation

    rho = results.density_matrix_of(Qs)
    pFail = rho[-1,-1]
    pSucc = 1-pFail

    trace = 2*pSucc - 1

    return trace


def ExactEnergy(results, q1):
    # Calulcate energy using density matrix simulation
    
    h = np.array([  # We always measure the result in the Z basis
        [1,0],
        [0,-1]
    ])

    rho = results.density_matrix_of([q1])
    E = np.trace(rho @ h)

    return E


def SampledTraceCM( results, inv_cm ):
    # Calculate the trace from sampled results using data from the confusion matrix (cm)
    # Estimate the excess number of 11s that are measured because of noise using cm
    # We pass into this an inverted confsion_matrix to avoid multiple inversions

    N = sum(results.values())
    results_vector = np.array([
            results[0], results[1], results[2], results[3]
    ])

    PFail = np.dot(inv_cm , results_vector)[3] / N 

    trace = 1-2*PFail

    return trace


def SampleEnergyCorrected(results, invCM):
    # Calculate energy from sampled data, assuming everything is in Z basis.
    # Correct using the inversted 2x2 confusion matrix

    result_vector = np.array([results[0],results[1]])

    corrected_vector = np.dot(invCM, result_vector)

    num0s = corrected_vector[0]
    num1s = corrected_vector[1]

    E = ( num0s - num1s ) / (num0s + num1s)

    return E




#######################
# Calculate Confusion Matrices
#######################


def CMFromResultsTwoQ(R00,R01,R10,R11):
    ConfMatrix = np.zeros([4,4])
    N = sum(R00.values())  # assume they will all have the same shot nums

    for row, res in enumerate([R00,R01,R10,R11]):
        for col in range(4):
            ConfMatrix[row, col] = res[col]

    return ConfMatrix / N # Normalize values


def CMFromResultsOneQ(R0, R1):
    ConfMatrix = np.zeros([2,2])
    N = sum(R0.values())

    for row, res in enumerate([R0,R1]):
        for col in range(2):
            ConfMatrix[row, col] = res[col]

    return ConfMatrix / N 

def CMFromResults(Res):
    m = len(Res)
    ConfMatrix = np.zeros([m,m])
    N = sum(Res[0].values())

    for row, res in enumerate(Res):
        for col in range(m):
            ConfMatrix[col, row] = res[col]

    return ConfMatrix / N


