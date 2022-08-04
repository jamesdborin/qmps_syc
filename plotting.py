import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_fig2a():

    # data comes from the gs_traces (see how to plot below)
    # classical data comes from the fiels reference_gs_energy.csv and tfim_exact_gs.npy

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
        plt.errorbar( data['g'][i], data['OnChipEnergy'][i], yerr = data['OnChipStd'][i], c=cmap(1) )
        
        plt.errorbar( data['g'][i], data['OnChipRescale'][i], yerr = data['RescaleStd'][i], c=cmap(2) )
    

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
    plt.show()


def plot_fig2a_inset():
    data = {
        'g': [0.2,0.4,0.6,0.8,1.0,1.2,1.4],
        'ExactEnergy': [ -1.09,-1.16,-1.25,-1.376984, -1.6326 ],
        'OnChipEnergy': [ -0.922,-0.948,-1.012,-1.086,-1.189,-1.316,-1.444 ],
        'OnChipStd': [ 0.03,0.025,0.015,0.016,0.013,0.016,0.017 ],
        'RescaleStd': [ 0.033,0.025,0.016,0.016,0.013,0.017,0.018 ],
        'OnChipRescale':[ -1.019,-0.999,-1.067,-1.134,-1.251,-1.352,-1.484 ],
        'Measured' : [ True,True, True, True, True, True, True ]
    }

    fsize=15

    classical_es = '/home/jamie/Dropbox/QmpSyc/qmpsyc/data/ground_state/classical_gs/reference_gs_energy.csv'

    classical_df = pd.read_csv(classical_es)
    gs = classical_df['g']
    es = classical_df['Energy']

    exact_data = '/home/jamie/Dropbox/QmpSyc/tfim_exact_gs.npy'
    exact_df = np.load(exact_data)

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


def plot_fig2b2c(csv_file, save_file):
    # data to plot the traces as seen in fig 2b and 2c
    data = pd.read_csv(csv_file)
    print(data.head())

    g_change = data['Iteration'] == 1
    gs = data['g'][g_change]

    change_locs = gs.index.to_list()
    change_gs = gs.to_list()
    
    cmap = plt.get_cmap("tab10")

    plt.plot(data['RescaledEnergy'], c=cmap(0),  label = 'Rescaled Energy')
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




if __name__ == '__main__':

    # here is how to plot fig2b, change the file to data/gs_data/gs_traces/Dec-14-2021RainbowRescaledAdiabaticSearchG04.csv or data/gs_data/gs_traces/Dec-16-2021RainbowRescaledAdiabaticSearchG04.csv to get equivalent data to 2c
    #   
    file = 'data/gs_data/gs_traces/Dec09RainbowRescaledAdiabaticSearchG12.csv'
    save = "g12_Dec09.pdf"
    plot_fig2b2c(file, save)