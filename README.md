# qmps_syc
This repository holds the code to generate results for the work "Simulating groundstate and dynamical quantum phase transitions on a superconducting quantum computer"

The files correspond to the following sections of the work:

Design and implementation of the translationally invariant ansatz: crcuits.py, energy.py, trace_distance.py

Analysis of the ground state data, and classical simulations: classical_tfim_ground_state.py, plotting.py. Data is stored in data/gs_data.

Calculating Overlaps Of Translationally Invariant States: Time Ev Figure Plotting.ipynb

Circuits For Time Evolution: circuits_tev.py

Numerics For Time Evolution: timeevolution.py, data_analysis.py, plot_time_series.py, overlaps_classical.py, overlaps_marginal.py. Data is stored in data/time_evo_data/

Some work cannot be recreated due to unavailability of the device. However some device-specific code has been included for reference, such as in floquet.py.
