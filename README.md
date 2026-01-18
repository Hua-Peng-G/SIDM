SIDM Halo Collapse Simulation in GR + hydrodynamical
Author: Hua-peng Gu
Date: January 2026

========================================================================
1. OVERVIEW
========================================================================

This program simulates the collapse of Self-Interacting Dark Matter (SIDM) 
halos under General Relativistic (GR) and hydrodynamical conditions. It 
combines the Misner-Sharp equations with a heat conduction model for SIDM. 

The theoretical foundation and numerical framework for this method is 
illustrated in: Gu, Jiang, and Chen (2026).

CITATION REQUIREMENT:
If you use this code or method in your research, please cite our paper:
Gu, Jiang, and Chen (2026).
Link: https://......

========================================================================
2. QUICK START
========================================================================

To start the simulation with the default NFW profile, simply run the 
following command:

    python main.py

The program will visualize the density profile evolution in real-time. 
Simulation results will be automatically saved to 'output.npy'.

========================================================================
3. CONFIGURATION AND INPUTS
========================================================================

You can customize the simulation by modifying the following text files:

  - parameter.txt: Contains physical parameters (e.g., Rs, M, sigma).
  - config.txt:    Contains runtime controls (e.g., dt, steps).

Input Data Files:
-----------------
The program requires two initial data files (provided by default):

  1. A.npy: 
     The Lagrangian grid coordinates.

  2. initial.npy: 
     A (11 x 1001) array containing the initial values of the physical 
     quantities on 1001 spatial grids. 
     
     The 11 channels correspond to: 
     U, R, rho, epsilon, P, w, e^phi, m, Gamma, eA, q

You may replace these files with custom profiles of your interest.

========================================================================
4. OUTPUT DATA
========================================================================

The results are saved in 'output.npy'. 

Data Shape: (N, 11, 1001)
  - N = (total_steps / save_interval) + 1
  - The second dimension (11) corresponds to the variables listed above.
  - The third dimension (1001) corresponds to the spatial grids.

========================================================================
5. UNITS AND SCALING
========================================================================

The code operates in natural units using the following scaling factors:

  Scale_R     = Rs / (GM/c^2)
  Scale_sigma = M / (GM/c^2)^2

Units for Input/Output Variables:
---------------------------------

  Variable       Unit
  --------       ----
  U              Scale_R * c
  R              Rs
  rho            rho_s / 18.7
  epsilon        Scale_R * c^2
  P              (rho_s / 18.7) * Scale_R * c^2
  w              1 (Dimensionless)
  e^phi          1 (Dimensionless)
  m              M (Halo Mass)
  Gamma          1 (Dimensionless)
  q              Scale_R^(13/2) / Scale_sigma * (c^9 * G^(-3) * M^(-2))

  Note: 'eA' is a temporary auxiliary variable.