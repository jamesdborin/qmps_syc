import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def import_dummy_data():

    e = 0.1

    exact_vals = np.random.rand(8)
    mean_vals = exact_vals + e * np.random.rand(8) 
    std_vals = e * np.random.rand(8)

    return exact_vals, mean_vals, std_vals


def import_data(step):

    exact_smooth = pd.read_csv('/home/jamie/Dropbox/QmpSyc/qmpsyc/data/time_evo_data/classical_interpolated_smooth_step/exact_smooth - data_0.csv')

    exact = pd.read_csv('/home/jamie/Dropbox/QmpSyc/qmpsyc/images/time_evo_cost_functions/IdentityRightEnvironmentRatio/Exact.csv')

    exper = pd.read_csv('/home/jamie/Dropbox/QmpSyc/qmpsyc/images/time_evo_cost_functions/IdentityRightEnvironmentRatio/Experiment.csv')

    stds = pd.read_csv('/home/jamie/Dropbox/QmpSyc/qmpsyc/images/time_evo_cost_functions/IdentityRightEnvironmentRatio/Stds.csv')

    x_smooth = exact_smooth['step']

    return exact[step], exper[step], stds[step], x_smooth, exact_smooth[step]

def plot_data(exact, mean, stds,x_smooth, exact_smooth):
    x = np.arange(8)

    plt.scatter(x, exact, c = 'b')
    plt.plot(x_smooth,exact_smooth, 'b', label = 'Exact Value')

    measured = plt.errorbar(x, mean, stds, c='g', linestyle = '--', marker = 'x', label = "Measured Value")

    # plt.legend(fontsize = 20)
    plt.xlabel('Parameter Step',fontsize = 22)
    plt.ylabel('Rescaled Measured Overlap', fontsize = 22)
    plt.ylim(0,1.1)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
        

def main():
    for i in range(5):
        a=i
        b=i+1
        e,m,s,x_s,e_s = import_data(f'{a}->{b}')
        plot_data(e,m,s,x_s,e_s)
        plt.savefig(f'/home/jamie/Dropbox/QmpSyc/qmpsyc/images/time_evo_cost_functions/IdentityRightEnvironmentRatio/{a}_{b}_smooth.pdf')
        plt.show()


if __name__ == '__main__':
    main()