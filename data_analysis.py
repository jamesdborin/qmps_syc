import pickle
from qmpsyc.circuits_tev import analyse_probs_single, analyse_probs_post_select
import numpy as np
import csv
from collections import defaultdict

def key_2_str(key):

    F,S,R,run = key
    str = f'({F},{S},{R},{run})'
    
    return str

def str_2_key(s:str):
    
    stripped_str = s.replace('(', '')
    stripped_str = stripped_str.replace(')', '')

    F,S,R,run = stripped_str.split(',')
    F = int(F)
    S = int(S)
    R = int(R)

    return (F,S,R,run)


def trace_out_sites():
    fname = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/all_raw_data.pkl'

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    data_keys = data.keys()
    all_results = defaultdict(list)

    for k in data_keys:
        F,S,R,run = k
        new_key_1 = (F,S,R,run,'1site')
        new_key_2 = (F,S,R,run,'2site')
        
        for step in range(8):
            circuit1data, circuit2data = analyse_probs_single(data[k][step][0], 2)
            circuit1prob01, circuit1prob02 = circuit1data
            circuit2prob01, circuit2prob02 = circuit2data
            prob_1_res = np.mean([circuit1prob01, circuit2prob01])
            prob_2_res = np.mean([circuit1prob02, circuit2prob02])

            all_results[new_key_1].append(prob_1_res)
            all_results[new_key_2].append(prob_2_res)

    csv_filename = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data.csv'

    keys = all_results.keys()

    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[all_results[key] for key in keys]))


def analyse_single_site_post_select():
    fname = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/all_raw_data.pkl'

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    data_keys = data.keys()
    all_results = defaultdict(list)

    # perform the data analysiss to get the raw data
    for k in data_keys:
        F,S,R,run = k
        new_key_1 = (F,S,R,run,'1site')
        new_key_2 = (F,S,R,run,'2site')
        
        for step in range(8):
            circuit1data, circuit2data = analyse_probs_post_select(data[k][step][0], 2)
            circuit1prob01, circuit1prob02 = circuit1data
            circuit2prob01, circuit2prob02 = circuit2data
            prob_1_res = np.mean([circuit1prob01, circuit2prob01])
            prob_2_res = np.mean([circuit1prob02, circuit2prob02])

            all_results[new_key_1].append(prob_1_res)
            all_results[new_key_2].append(prob_2_res)
    
    # save the file with the raw data for reference
    csv_filename = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select.csv'

    keys = all_results.keys()

    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[all_results[key] for key in keys]))

    rescaling = {}

    # calculate rescalings:
    for k in all_results.keys():
        F,S,R,run,site = k
        
        is_1site_identity = (run == 'iden') and (site == '1site')
        
        if is_1site_identity:
            rescaling[k] = 1 / np.array( all_results[k] )

    # save the rescaling
    csv_filename_rescale = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_rescaling.csv'

    keys = rescaling.keys()

    with open(csv_filename_rescale, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[rescaling[key] for key in keys]))

    # calculate the rescaled values
    rescaled_1_site_values = {}

    for k in all_results.keys():
        F,S,R,run,site = k

        is_1site_res = (run == 'res') and (site == '1site')
        
        if is_1site_res:
            res_key = k
            rescale_key = (F,S,R,'iden','1site')

            rescaled_1_site_values[k] = np.array(all_results[res_key])  * rescaling[rescale_key]

    csv_filename_rescaled = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_rescaled_values.csv'

    keys = rescaled_1_site_values.keys()

    with open(csv_filename_rescaled, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[rescaled_1_site_values[key] for key in keys]))

    averaged_rescaled_1_site_values = {}
    # now average the values

    runs_order = [[0,1],[1,2],[2,3],[3,4],[4,5]]
    batch = [0,1,2,3,4]

    for F,S in runs_order:
        result = np.zeros(8)
        
        for b in batch:
            result_key = (F,S,b,'res','1site')
            result += rescaled_1_site_values[result_key]

        result /= 5

        averaged_key = (F,S,'res','1site')
        averaged_rescaled_1_site_values[averaged_key] = result

    csv_filename_averaged = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_averaged_rescaled_values.csv'

    keys = averaged_rescaled_1_site_values.keys()

    with open(csv_filename_averaged, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[averaged_rescaled_1_site_values[key] for key in keys]))


if __name__ == '__main__':

    # step 1: load the data from the pickle file (be EXTREMELY CAREFUL NOT TO OVERWRITE IT)
    # how the parse the data:
    # data = {
    #   (F,S,R,run): [step_1, step_2, ..., step_8]
    # }
    # F = First optimal step
    # S = Second optimal step
    # R = Batch Run
    # step_i = [[circuit1 p01, circuit1 p02], [circuit2 p01, circuit2 p02]]
    # circuit1 and circuit2 are for each of the circuits run on the device at each step,


    fname = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/all_raw_data.pkl'

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    data_keys = data.keys()
    all_results = defaultdict(list)

    # perform the data analysiss to get the raw data
    for k in data_keys:
        F,S,R,run = k
        new_key_1 = (F,S,R,run,'1site')
        new_key_2 = (F,S,R,run,'2site')
        
        for step in range(8):
            circuit1data, circuit2data = analyse_probs_post_select(data[k][step][0], 2)
            circuit1prob01, circuit1prob02 = circuit1data
            circuit2prob01, circuit2prob02 = circuit2data
            prob_1_res = np.mean([circuit1prob01, circuit2prob01])
            prob_2_res = np.mean([circuit1prob02, circuit2prob02])

            all_results[new_key_1].append(prob_1_res)
            all_results[new_key_2].append(prob_2_res)
    
    # save the file with the raw data for reference
    csv_filename = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_squared.csv'

    keys = all_results.keys()

    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[all_results[key] for key in keys]))

    rescaling = {}

    # calculate rescalings:
    for k in all_results.keys():
        F,S,R,run,site = k
        
        is_2site_identity = (run == 'iden') and (site == '2site')
        
        if is_2site_identity:
            rescaling[k] = 1 / np.array( all_results[k] )

    # save the rescaling
    csv_filename_rescale = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_rescaling_squared.csv'

    keys = rescaling.keys()

    with open(csv_filename_rescale, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[rescaling[key] for key in keys]))

    # calculate the rescaled values
    rescaled_2_site_values = {}

    for k in all_results.keys():
        F,S,R,run,site = k

        is_2site_res = (run == 'res') and (site == '2site')
        
        if is_2site_res:
            res_key = k
            rescale_key = (F,S,R,'iden','2site')

            rescaled_2_site_values[k] = np.array(all_results[res_key])  * rescaling[rescale_key]

    csv_filename_rescaled = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_squared_rescaled_values.csv'

    keys = rescaled_2_site_values.keys()

    with open(csv_filename_rescaled, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[rescaled_2_site_values[key] for key in keys]))

    averaged_rescaled_2_site_values = {}
    # now average the values
    std_rescaled_2_site_values = {}

    runs_order = [[0,1],[1,2],[2,3],[3,4],[4,5]]
    batch = [0,1,2,3,4]

    for F,S in runs_order:
        result = np.zeros(8)
        result_sqrd = np.zeros(8)

        for b in batch:
            result_key = (F,S,b,'res','2site')
            result += np.sqrt(rescaled_2_site_values[result_key])

            # calculate the mean of the squares
            result_sqrd += np.abs(rescaled_2_site_values[result_key])

        result /= 5
        result_sqrd /= 5

        # std = sqrt (mean of squares - square of mean)
        std_dev = np.sqrt( result_sqrd - result**2 )


        averaged_key = (F,S,'res','2site')
        averaged_rescaled_2_site_values[averaged_key] = result
        
        std_dev_key = (F,S,'res','2site')
        std_rescaled_2_site_values[std_dev_key] = std_dev

    csv_filename_averaged = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_averaged_sqrt_squared_rescaled_values.csv'

    keys = averaged_rescaled_2_site_values.keys()

    with open(csv_filename_averaged, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[averaged_rescaled_2_site_values[key] for key in keys]))

    csv_filename_std = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/data/analysed_data_post_select_std_sqrt_squared_rescaled_values.csv'

    keys = std_rescaled_2_site_values.keys()

    with open(csv_filename_std, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[std_rescaled_2_site_values[key] for key in keys]))
