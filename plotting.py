import matplotlib.pyplot as plt
import json
import pandas as pd
import csv
from collections import defaultdict
import numpy as np
import sys
def plot_ground_state_results(filename):

    with open(filename,'r') as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize = (15,15))

    trace_results = data['Trace']
    tfim_results = data['TFIM']

    for name, (avg, std) in tfim_results.items():
        ax1.errorbar([name], [avg], yerr = std, label = name, ms = 10, marker = '.')
    
    ax1.hlines(y = tfim_results['Exact'][0], xmin = 0, xmax = 7, linestyles='--')
    ax1.set_ylim(-1.5, -0.8)
    ax1.title.set_text('TFIM Energy')


    for name, (avg, std) in trace_results.items():
        ax2.errorbar([name], [avg], yerr = std, label = name, ms = 10, marker = '.')

    ax2.hlines(y = 0, xmin = 0, xmax = 7, linestyles='--')
    ax2.title.set_text('Trace Distance')

    plt.tight_layout()
    ax1.legend()
    plt.show()


def plot_traces(energy_file, traces_file, exact_e_file, exact_t_file, rescaled_e):
    es = np.load(energy_file)
    ts = np.load(traces_file)
    eEs = np.load(exact_e_file)
    eTs = np.load(exact_t_file)
    rescale_eTs = np.load(rescaled_e)


    plt.plot(rescale_eTs, 'g', linestyle = '--', label = 'Rescaled Energy')
    plt.plot(es, 'r', label='Energy')
    plt.plot(ts, 'b', label = 'Trace Distance')
    plt.plot(eEs, 'r', linestyle = '--', label = 'Exact Energy')
    plt.plot(eTs, 'b', linestyle = '--', label = 'Exact Energy')


    plt.axhline(y = 0, xmin=0, xmax = 100, linestyle = '--')
    plt.axhline(y = -1.27, xmin=0, xmax = 100, linestyle = '--')  #-1.27 for J=-1,g=1
    plt.legend()
    plt.show()


def plot_csv(csv_file):
    
    data = pd.read_csv(csv_file)
    print(data.head())

    g_change = data['Iteration'] == 1
    gs = data['g'][g_change]

    change_locs = gs.index.to_list()
    change_gs = gs.to_list()

    plt.plot(data['RescaledEnergy'], 'g', linestyle = '--', label = 'Rescaled Energy')
    plt.plot(data['Energy'], 'r', label='Energy')
    plt.plot(data['Trace'], 'b', label = 'Trace Distance')
    plt.plot(data['ExactEnergy'], 'r', linestyle = '--', label = 'Exact Energy')
    plt.plot(data['ExactTrace'], 'b', linestyle = '--', label = 'Exact Energy')


    plt.axhline(y = 0, xmin=0, xmax = 100, linestyle = '--')
    plt.axhline(y = -1.6326, xmin=0, xmax = 100, linestyle = '--')  #-1.27 for J=-1,g=1
    
    for loc, g in zip(change_locs, change_gs):
        plt.axvline( x = loc, ymin = 0, ymax = 1, linestyle = ':' )
        plt.text( x = loc, y = -0.4, s = g )
    plt.legend()
    plt.show()


def paper_plot_2_dummy():
    # data = {
    #     'g': [0.2,0.4,0.6,0.8,1.0,1.2,1.4],
    #     'ExactEnergy': [ -1.09,-1.16,-1.25,-1.376984, -1.6326 ],
    #     'OnChipEnergy': [-0.9217,-0.9208, -1.03, -1.1006033024756792, -1.186801486441925, -1.325075365, -1.451189613 ],
    #     'OnChipRescale':[-1.0189, -0.9803, -1.085, -1.1487609412321418, -1.2452572447713446, -1.362479037, -1.488258965 ],
    #     'Measured' : [True,True, True, True, True, True, True ]
    # }

    data = {
        'g': [0.2,0.4,0.6,0.8,1.0,1.2,1.4],
        'ExactEnergy': [ -1.09,-1.16,-1.25,-1.376984, -1.6326 ],
        'OnChipEnergy': [ -0.922,-0.948,-1.012,-1.086,-1.189,-1.316,-1.444 ],
        'OnChipStd': [ 0.03,0.025,0.015,0.016,0.013,0.016,0.017 ],
        'RescaleStd': [ 0.033,0.025,0.016,0.016,0.013,0.017,0.018 ],
        'OnChipRescale':[ -1.019,-0.999,-1.067,-1.134,-1.251,-1.352,-1.484 ],
        'Measured' : [ True,True, True, True, True, True, True ]
    }


    classical_es = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/ground_state/classical_gs/reference_gs_energy.csv'

    classical_df = pd.read_csv(classical_es)
    gs = classical_df['g']
    es = classical_df['Energy']
    
    

    exact_data = '/home/jamie/Dropbox/QmpSyc/tfim_exact_gs.npy'
    exact_df = np.load(exact_data)

    cmap = plt.get_cmap("tab10")
    for i in range(len(data['g'])):
        #plt.scatter( data['g'][i], data['ExactEnergy'][i], c='b', marker = 'o' )
        # plt.scatter( data['g'][i], data['OnChipEnergy'][i], c='r', marker = 'o' )
        # plt.scatter( data['g'][i], data['OnChipRescale'][i], c='g', marker = 'o' )
        # plt.scatter( data['g'][i], data['OnChipEnergy'][i], c='r', marker = 'o' )
        plt.errorbar( data['g'][i], data['OnChipEnergy'][i], yerr = data['OnChipStd'][i], c=cmap(1) )
        
        # plt.scatter( data['g'][i], data['OnChipRescale'][i], c='g', marker = 'x' )
        plt.errorbar( data['g'][i], data['OnChipRescale'][i], yerr = data['RescaleStd'][i], c=cmap(2) )
    
    # for g,e in zip(gs, es):
    #     plt.scatter( g,e, c='b', marker = 'o')

    #plt.plot( data['g'], data['ExactEnergy'], 'b', label = 'Exact GS In Ansatz' )


    plt.plot(gs, es, c=cmap(0), label = 'Exact GS In Ansatz' )
    plt.plot(gs, exact_df, c=cmap(4), label = "Exact GS")

    plt.plot( data['g'], data['OnChipEnergy'], c = cmap(1), label = 'Measured GS Energy', marker = '^', ms=6 )
    plt.plot( data['g'], data['OnChipRescale'], c=cmap(2),label = 'Rescaled GS Energy', marker='o')
    fsize=15
    plt.legend(loc=1, fontsize=fsize)
    plt.xlabel('g', fontsize=fsize)
    plt.ylabel('Energy',fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    #plt.title('Energy of Optimized GS: Exact vs Measured vs Rescaled') 
    #plt.savefig("/home/jamie/Dropbox/QmpSyc/qmpsyc/images/GSTraces/GSPlotTest.pdf")
    plt.show()

    error = []
    for i,g in enumerate(data['g']):
        exact_e = classical_df[classical_df['g'] == g]['Energy']
        error.append(np.abs(exact_e - data['OnChipRescale'][i]))

    plt.plot(data['g'], error)
    plt.yscale('log')
    plt.ylim(0.0001,0.1)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylabel("Energy Error",fontsize=fsize)
    plt.xlabel("g",fontsize=fsize)
    plt.tight_layout()
    plt.savefig("/home/jamie/Dropbox/QmpSyc/qmpsyc/images/GSTraces/GSLogError.pdf")
    plt.show()

def average_last_N(file,N):
    df = pd.read_csv(file)

    es = df['Energy'].to_numpy()
    r_es = df['RescaledEnergy'].to_numpy()

    averaged_e = np.mean(es[-N:])
    averaged_r_e = np.mean(r_es[-N:])

    return averaged_e, averaged_r_e


def plot_csv_no_gs(csv_file, save_file):
    
    data = pd.read_csv(csv_file)
    print(data.head())

    g_change = data['Iteration'] == 1
    gs = data['g'][g_change]

    change_locs = gs.index.to_list()
    change_gs = gs.to_list()
    
    cmap = plt.get_cmap("tab10")

    plt.plot(data['RescaledEnergy'], c=cmap(0),  label = 'Rescaled Energy')
    #plt.plot(data['Energy'], 'r', label='Energy')
    plt.plot(data['Trace'], c=cmap(1), label = 'Trace Distance')
    plt.plot(data['ExactEnergy'], c=cmap(3), linestyle = '--', label = 'Exact Energy')
    plt.plot(data['ExactTrace'], c=cmap(2), linestyle = '--', label = 'Exact Trace')

    plt.axhline(y = 0, xmin=0, xmax = 100, linestyle = '--')
    
    fsize = 15
    for loc, g in zip(change_locs, change_gs):
        plt.axvline( x = loc, ymin = 0, ymax = 1, linestyle = ':' )
        plt.text( x = loc, y = -0.4, s = g, fontsize=fsize )
    plt.legend(fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel("Iterations", fontsize = fsize)
    plt.ylabel("Energy", fontsize = fsize)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


def plot_all_and_save():
    dir = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/opt_traces/'
    save_dir = "/home/jamie/Dropbox/QmpSyc/qmpsyc/images/GSTraces/"

    fnames = ["Dec-15-2021RainbowRescaledAdiabaticSearchG02",
    "Dec-14-2021RainbowRescaledAdiabaticSearchG04",
    "Dec-16-2021RainbowRescaledAdiabaticSearchG04",
    "Dec02RainbowRescaledAdiabaticSearchG06",
    "Dec05RainbowRescaledAdiabaticSearchG08",
    "Dec-14-2021RainbowRescaledAdiabaticSearchG08_2",
    "Dec05RainbowRescaledAdiabaticSearchG10",
    "Dec06RainbowRescaledAdiabaticSearchG10",
    "Dec07RainbowRescaledAdiabaticSearchG10Cont",
    "Dec07RainbowRescaledAdiabaticSearchG10Cont2",
    "Dec08RainbowRescaledAdiabaticSearchG12",
    "Dec09RainbowRescaledAdiabaticSearchG12",
    "Dec12RainbowRescaledAdiabaticSearchG14",
    "Dec-13-2021RainbowRescaledAdiabaticSearchG14",
    "Dec-13-2021RainbowRescaledAdiabaticSearchG14cont"
    ]

    snames = [
        "G02Dec15",
        "G04Dec14",
        "G04Dec16",
        "G06Dec02",
        "G08Dec05",
        "G08Dec14",
        "G10Dec05",
        "G10Dec06",
        "G10Dec07",
        "G10Dec07_2",
        "G12Dec08",
        "G12Dec09",
        "G14Dec12",
        "G14Dec13",
        "G14Dec13_2"
    ]

    for f,s in zip([fnames[11]],[snames[11]]):
        csv_file = dir + f + '.csv'
        save_file = save_dir + s + '.pdf'
        plot_csv_no_gs(csv_file, save_file)


def get_run_statistics():
    idx = int(sys.argv[1])
    last_N = int(sys.argv[2])

    fnames = [
    # "Dec-15-2021RainbowRescaledAdiabaticSearchG02", 
    # "Dec-14-2021RainbowRescaledAdiabaticSearchG04",
    # "Dec-16-2021RainbowRescaledAdiabaticSearchG04",
    # "Dec02RainbowRescaledAdiabaticSearchG06",
    # "Dec05RainbowRescaledAdiabaticSearchG08",
    # "Dec-14-2021RainbowRescaledAdiabaticSearchG08_2",
    # "Dec05RainbowRescaledAdiabaticSearchG10",
    # "Dec06RainbowRescaledAdiabaticSearchG10",
    # "Dec07RainbowRescaledAdiabaticSearchG10Cont",
    # "Dec07RainbowRescaledAdiabaticSearchG10Cont2",
    "Dec08RainbowRescaledAdiabaticSearchG12",
    "Dec09RainbowRescaledAdiabaticSearchG12",
    # "Dec12RainbowRescaledAdiabaticSearchG14",
    # "Dec-13-2021RainbowRescaledAdiabaticSearchG14",
    # "Dec-13-2021RainbowRescaledAdiabaticSearchG14cont"
    ]

    dir = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/opt_traces/'
    csv_file = dir + fnames[idx] + '.csv'

    data = pd.read_csv(csv_file)
    
    es = data['RescaledEnergy']

    data_es = es[-last_N:]

    AVG = np.mean(data_es)
    MED = np.median(data_es)
    STD = np.std(data_es)
    MAX = np.max(data_es)
    MIN = np.min(data_es)
    END = data_es.to_list()[-1]

    res_str = f"""
    
    avg = {AVG}
    med = {MED}
    std = {STD}
    max = {MAX}
    min = {MIN}
    end = {END}
    """
    print(res_str)



if __name__ == '__main__':
#Dec-15-2021RainbowRescaledAdiabaticSearchG02.csv
    #plot_all_and_save()
    #paper_plot_2_dummy()
    
    file = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/opt_traces/Dec09RainbowRescaledAdiabaticSearchG12.csv'
    save = "/home/jamie/Dropbox/QmpSyc/qmpsyc/images/GSTraces/g12_Dec09_Recolored.pdf"
    plot_csv_no_gs(file, save)