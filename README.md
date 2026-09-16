# SIDM Halo Collapse Simulation in GR + Hydrodynamical

**Author:** Hua-peng Gu  
**Date:** September 2026

## 1. Overview

This program simulates the collapse of Self-Interacting Dark Matter (SIDM) halos under General Relativistic (GR) and hydrodynamical conditions. It combines the Misner-Sharp equations with a heat conduction model for SIDM.

The theoretical foundation and numerical framework for this method are illustrated in: **Gu, Jiang, Chen et al. (2026a)**. In **Gu, Jiang, Chen et al. (2026b)**, we upgrade it to follow the dark accretion as well. Unlike conventional fluid treatments of SIDM halos, this extension
allows us to capture BH seed formation and follow the ensuing accretion within a single, unified framework.

> [IMPORTANT]
> **CITATION REQUIREMENT**
> If you use this code or method in your research, please cite our papers:  
> **Gu, Jiang, Chen et al. (2026a)**
>
> [doi : 10.1103/6sl3-kjzf]
>
> **Gu, Jiang, Chen et al. (2026b)**
>
> [arXiv:2609.16122]

---

## 2. Quick Start

To start the simulation with the default NFW profile, simply run the following command:

    python main.py

The program will visualize the density profile evolution in real-time. Simulation results will be automatically saved to `output.npy`.

---

## 3. Configuration and Inputs

You can customize the simulation by modifying the following text files:

- `parameter.txt`: Contains physical parameters (e.g., $R_s$, $M$, $\sigma$).
- `config.txt`: Contains runtime controls (e.g., `dt`, `steps`).

### Input Data Files

The program requires two initial data files (provided by default):

1. **`A.npy`**
   The Lagrangian grid coordinates.

2. **`initial.npy`**
   A `(11 x 251)` array containing the initial values of the physical quantities on 251 spatial grids.  
   The 11 channels correspond to:  
   `U`, `R`, `rho`, `epsilon`, `P`, `w`, `e^phi`, `m`, `Gamma`, `eA`, `q`

You may replace these files with custom profiles of your interest.

---

## 4. Output Data

The results are saved in `output.npy`.

* **Data Shape:** `(N, 11, 251)`
  * `N = (total_steps / save_interval) + 1`
  * The second dimension (`11`) corresponds to the variables listed above.
  * The third dimension (`251`) corresponds to the spatial grids.

---

## 5. Units and Scaling

The code operates in natural units using the following scaling factors:

$$R_{scale} = \frac{R_s}{GM/c^2}$$

$$\sigma_{scale} = \frac{M}{(GM/c^2)^2}$$

### Units for Input/Output Variables

| Variable    | Unit                                                         |
| :---------- | :----------------------------------------------------------- |
| **U**       | $R_{scale} \cdot c$                                          |
| **R**       | $R_s$                                                        |
| **rho**     | $\rho_s / 18.7$                                              |
| **epsilon** | $R_{scale} \cdot c^2$                                        |
| **P**       | $(\rho_s / 18.7) \cdot R_{Scale} \cdot c^2$                  |
| **w**       | 1 (Dimensionless)                                            |
| **e^phi**   | 1 (Dimensionless)                                            |
| **m**       | $M$ (Halo Mass)                                              |
| **Gamma**   | 1 (Dimensionless)                                            |
| **q**       | $R_{scale}^{13/2} / \sigma_{scale} \cdot (c^9 G^{-3} M^{-2})$ |

> **Note:** `eA` is a temporary auxiliary variable.
