import cirq
import numpy as np
from scipy.linalg import expm
from qmps.ground_state import Hamiltonian
import matplotlib.pyplot as plt
from tqdm import tqdm

from qmpsyc.circuits_tev import HalfTrotterGenerator
from qmpsyc.circuits import StateAnsatzXZ
from qmpsyc.time_evolution import analytic_tev_half_trotter_overlap

from loschmidt import times_fisher_zero

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
    max_time = 5.0
    g1 = 0.2
    t_peaks = times_fisher_zero(1.5, 0.2, max_time)
    dtau = t_peaks[0] / 5

    θ0 = np.load('d0xzstate2_params_g1.5_tol1e-05.npy', allow_pickle=True)
    θA = np.load('d0xzstate2_params_g1.5_tol1e-05.npy', allow_pickle=True)

    results_dir = '/home/vinulw/Dropbox/project-time_ev_google/291221_Opt_Params/half_trotter/'
    params = np.load(results_dir + '291221165112_ls_param_log_stateansatzxz2_half_trotter.npy')
    params = np.random.rand(4, len(params[0]))

#    t0 = 1
#    t1 = 2
#    θ0 = params[t0]
#    θA = params[t1]
#
    U1 = expm(-1j*Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()*dtau*2)
#    n = 2
#
#    exact = analytic_tev_half_trotter_overlap(θ0, θA, U1, Ansatz=StateAnsatzXZ) ** 2
#
#    probs = []
#    probs_exact = []
#    n_range = list(range(1, 8))
#    qubits = cirq.LineQubit.range(max(n_range)*2 + 1)
#    for n in tqdm(n_range):
#        qubits_ = qubits[:2*n+1]
#        gen = HalfTrotterGenerator(U1, n, qubits=qubits_, Ansatz=StateAnsatzXZ)
#
#        ov_circuit = gen.overlap_circuit(θ0, θA)
#        prob_exact = gen.prob0_overlap_exact(θ0, θA)
#        probs_exact.append(prob_exact)
#        ov_circuit.append(cirq.measure(*qubits_, key = 'meas'), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
#        sim = cirq.Simulator()
#
#        results = sim.run(ov_circuit, repetitions = 10000)
#        res = results.histogram(key='meas')
#
#        p0 = res[0] / sum(res.values())
#        probs.append(p0)
#
#
#
#    # Calculate trace probs
#    n = 6
#    qno = 2*n + 1
#    qnop1 = 2*(n+1) + 1
#    qubits_ = qubits[:qno]
#    qubitsp1_ = qubits[:qnop1]
#    gen = HalfTrotterGenerator(U1, n+1, qubitsp1_, Ansatz=StateAnsatzXZ)
#    ov_circuit = gen.overlap_circuit(θ0, θA)
#    ov_circuit.append(cirq.measure(*qubits_, key = 'meas'), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
#
#    sim = cirq.Simulator()
#
#    results = sim.run(ov_circuit, repetitions = 10000)
#    res = results.histogram(key='meas')
#
#    gen2 = HalfTrotterGenerator(U1, n, qubits[:qno + 1], Ansatz=StateAnsatzXZ)
#    ov_circuit2 = gen2.overlap_circuit_p1(θ0, θA)
#    ov_circuit2.append(cirq.measure(*qubits_, key = 'meas2'), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
#
#    sim = cirq.Simulator()
#
#    results = sim.run(ov_circuit2, repetitions = 10000)
#    res2 = results.histogram(key='meas2')
#
#    #for k in res.keys():
#    #    print('Key: {}'.format(k))
#    #    print('     Binary: {}'.format(format(k, 'b')))
#
#
#    probs_same = []
#    for i in range(1, n):
#        excluded_indices = range(2*i + 1, qno)
#        res_n = calculate_marginals(res, qno, excluded_indices)
#        p0n = res_n[0] / sum(res_n.values())
#        probs_same.append(p0n)
#
#    probs_same.append(res[0] / sum(res.values()))
#    print(probs_same)
#
#    probs_same2 = []
#    for i in range(1, n):
#        excluded_indices = range(2*i + 1, qno)
#        res_n = calculate_marginals(res2, qno, excluded_indices)
#        p0n = res_n[0] / sum(res_n.values())
#        probs_same2.append(p0n)
#
#    probs_same2.append(res2[0] / sum(res2.values()))
#
#    probs = np.array(probs)
#    probs_exact = np.array(probs_exact)
#    probs_same = np.array(probs_same)
#    probs_same2 = np.array(probs_same2)
#
#    overlaps = probs[1:] / probs[:-1]
#    overlaps_exact = probs_exact[1:] / probs_exact[:-1]
#    overlaps_same = probs_same[1:] / probs_same[:-1]
#    overlaps_same2 = probs_same2[1:] / probs_same2[:-1]
#
#    plt.plot(n_range[:-1], overlaps, 'x--', label='noiseless sim')
#    plt.plot(n_range[:-1], overlaps_exact, 'x--', label='exact')
#    plt.plot(range(1, len(overlaps_same) + 1), overlaps_same, 'x--', label='single run extra site')
#    plt.plot(range(1, len(overlaps_same2) + 1), overlaps_same2, 'x--', label='single run extra state')
#    plt.axhline(exact, ls='--')
#    plt.title('Params t={} & t={}'.format(t0, t1))
#    plt.legend()

    n = 5
    qno = 2*n + 1
    qubits = cirq.LineQubit.range(qno + 1)
    qubits_ = qubits[:-1]
    repeats = 5

    ov_means = []
    ov_stds = []
    ov_means_ps = []
    ov_stds_ps = []
    analytic_overlaps = []

    ov_machine_means = []
    ov_machine_stds = []

    gen = HalfTrotterGenerator(U1, n, qubits, Ansatz=StateAnsatzXZ)

    for θ0, θA in zip(params[:-1], params[1:]):
        analatic_overlap =  analytic_tev_half_trotter_overlap(θ0, θA, U1, StateAnsatzXZ) ** 2

        results = run_noiseless_params(θ0, θA, repeats=5)

        overlaps_lst = analyse_overlaps_single(results, n)
        overlaps_lst_ps = analyse_overlaps_post_select(results, n)
        results_lst = analyse_probs_single(results, n)

        ov_mean = np.mean(overlaps_lst, axis=0)
        ov_std = np.std(overlaps_lst, axis=0)
        ov_mean_ps = np.mean(overlaps_lst_ps, axis=0)
        ov_std_ps = np.std(overlaps_lst_ps, axis=0)

        print('Analytic overlap: {}'.format(analatic_overlap))
        print('Noiseless overlaps: {} +- {}'.format(ov_mean, ov_std))
        print('Noiseless overlaps ps: {} +- {}'.format(ov_mean_ps, ov_std_ps))
        print('Results list: {}'.format(results_lst))
        ov_means.append(ov_mean)
        ov_stds.append(ov_std)
        ov_means_ps.append(ov_mean_ps)
        ov_stds_ps.append(ov_std_ps)
        analytic_overlaps.append(analatic_overlap)


#    ov_means_machine = []
#    ov_stds_machine = []
#
#    for i, (θ0, θA) in enumerate(zip(params[:-1], params[1:])):
#        print('Machine run {}'.format(i))
#
#        gen = HalfTrotterGenerator(U1, n, qubits, Ansatz=StateAnsatzXZ)
#        results = gen.overlap_machine_qubits_results(θ0, θA, shots=50000, repeats=3)
#
#        overlaps_lst = []
#        for result in results:
#            overlaps_lst.extend(analyse_overlaps_single(result, n))
#
#        ov_mean = np.mean(overlaps_lst, axis=0)
#        ov_std = np.std(overlaps_lst, axis=0)
#        print('Analytic overlap: {}'.format(analytic_overlaps[i]))
#        print('Noiseless overlaps: {} +- {}'.format(ov_mean, ov_std))
#        ov_means_machine.append(ov_mean)
#        ov_stds_machine.append(ov_std)

    plt.figure()

    ov_means = np.array(ov_means)
    ov_stds = np.array(ov_stds)
    ov_means_ps = np.array(ov_means_ps)
    ov_stds_ps = np.array(ov_stds_ps)

#    ov_means_machine = np.array(ov_means_machine)
#    ov_stds_machine = np.array(ov_stds_machine)

    x_range = np.arange(len(analytic_overlaps))
    for i in range(len(ov_means[0])):
        print('Currently plotting {}'.format(i))
        print(ov_means[:, i])
        plt.errorbar(x_range, ov_means[:, i], yerr=ov_stds[:, i], marker='x', ls='--', label='noiseless {}'.format(i))
        plt.errorbar(x_range, ov_means_ps[:, i], yerr=ov_stds_ps[:, i], marker='x', ls='--', label='noiseless ps {}'.format(i))

    plt.plot(x_range, analytic_overlaps, 'x--', label='analytic')
    plt.title('Trace method, n={}'.format(n))
    plt.legend()
    plt.show()



    #for j, (θ0, θA) in enumerate(zip(params[:-1], params[1:])):
    #    print('Step {}'.format(j))
    #    gen = HalfTrotterGenerator(U1, n, qubits, Ansatz=StateAnsatzXZ)
    #    ov_circuit = gen.overlap_circuit_p1(θ0, θA)
    #    ov_circuit.append(cirq.measure(*qubits_, key = 'meas'), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    #
    #
    #    ov_noiseless = []
    #    for _ in repeats:
    #        sim = cirq.Simulator()
    #
    #        results = sim.run(ov_circuit, repetitions = 100000)
    #        res = results.histogram(key='meas')
    #
    #        probs_same = []
    #        for i in range(1, n):
    #            excluded_indices = range(2*i + 1, qno)
    #            res_n = calculate_marginals(res, qno, excluded_indices)
    #            p0n = res_n[0] / sum(res_n.values())
    #            probs_same.append(p0n)
    #
    #        probs_same.append(res[0] / sum(res.values()))
    #        probs_same = np.array(probs_same)
    #    overlaps_same = probs_same[1:] / probs_same[:-1]
    ##    np.save(save_path + 'noiseless_{}'.format(j), overlaps_same)
    ##    overlaps_same = np.load(save_path + 'noiseless_{}.npy'.format(j) , allow_pickle=True)
    #    for i, ov in enumerate(overlaps_same):
    #        noiseless_ov[i].append(ov)
    #
    #    res = gen.prob0_overlap_machine_results(θ0, θA, shots=50000, repeats=1)
    #    with open(save_path + 'rainbow_res_{}'.format(j), 'wb') as f:
    #        pickle.dump(res, f)
    #
    #    with open(save_path + 'rainbow_res_{}'.format(j), 'rb') as f:
    #        res = pickle.load(f)
    #    probs_machine = process_results_marginal(res, n, qno)
    #
    #    overlaps_machine = probs_machine[1:] / probs_machine[:-1]
    #    machine_ov.append(overlaps_machine)
    #    exact = analytic_tev_half_trotter_overlap(θ0, θA, U1, Ansatz=StateAnsatzXZ) ** 2
    #    analytic_ov.append(exact)
    #
    #
    #x = list(range(len(params) - 1))
    #print(noiseless_ov)
    #
    #plt.plot(x, analytic_ov, 'x--', label='analytic')
    #plt.plot(x, noiseless_ov[0], 'x--', label='noiseless 0')
    #plt.plot(x, noiseless_ov[1], 'x--', label='noiseless 1')
    #plt.legend()
    #plt.show()
