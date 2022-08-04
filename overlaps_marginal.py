import cirq
import numpy as np
from scipy.linalg import expm
from qmps.ground_state import Hamiltonian
import matplotlib.pyplot as plt
from tqdm import tqdm

from circuits_tev import HalfTrotterGenerator
from circuits import StateAnsatzXZ
from time_evolution import analytic_tev_half_trotter_overlap


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


def post_select_indices(bin_str, indices):
    '''
    Check if the string matches the post selection criteria
    '''
    for i in indices:
        if bin_str[i] != '0':
            return False
    return True


from qmpsyc.overlaps_optimisation import get_device, find_qubits
from qmpsyc.circuits_tev import process_results_marginal
import pickle
def run_noiseless_params(θ0, θA, repeats=1):
    results = []
    for _ in tqdm(range(repeats)):
        gen = HalfTrotterGenerator(U1, n, qubits, Ansatz=StateAnsatzXZ)
        ov_circuit = gen.overlap_circuit_p1(θ0, θA)
        ov_circuit.append(cirq.measure(*qubits_, key = 'meas'), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
        sim = cirq.Simulator()

        result = sim.run(ov_circuit, repetitions=100000)
        results.append(result.histogram(key='meas'))
    return results

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

def analyse_probs_post_select(results, n):
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

if __name__=="__main__":
    pass
