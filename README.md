# gii_calculator.py
Compute the Global Instability Index (GII) of a structure using bond valence parameters (tabulated or user-supplied). Includes:
- BVparams: helper class to look up default bond valence parameters from included .cif files  
- GIICalculator: class to calculate the GII of an oxidation state-decorated Pymatgen structure object (pymatgen.core.structure.Structure)
  - Note: This class will not work correctly for <= Pymatgen v. 2020.1.10, as PeriodicNeighbor objects are returned instead of NearestNeighbor objects   

# site_optimization.py
Adjust single site positions to minimize the GII. Includes:
- SiteClusterOptimization: class to optimize the Cartesian coordinates of a single Pymatgen site (pymatgen.core.sites.PeriodicSite) to minimize GII
- GIIMinimizer: class that minimizes the GII of a full Pymatgen structure object by iterative site optimization (includes different optimization strategies)

# parameterization.py
Parameterize R0 and B parameters using Pymatgen structure objects and user-supplied DFT energetics
- BVParamOptimization: class which optimizes a single cation/anion bond valence parameters (R0 and B) based on the oxidation-state assigned structures provided
- GeneralBVParamOptimizationOuterLoop: Manages the general bond valence parameter optimization of many different cation/anion bond valence parameters for a dataset of structures and DFT energies; supports gs_gii with Pearson correlation coefficient constraint and di2_rmsd parameterization
- CompositionSpecificBVParamOptimizationOuterLoop: Manages the composition-specific bond valence parameter optimization for datasets of structures and DFT energies *separated by composition*; supports gs_gii with Pearson correlation coefficient constraint and di2_rmsd parameterization

# visualization.py
Collection of plots to visualize the number of DFT ground states correctly ordered by GII, the number of total structures correctly ordered by GII, and Pearson coefficient distributions, as well as plots to compare different R0 parameters

# analysis.py
Supporting functions for visualization.py

# Dependencies
Python packages:
- Numpy >= v. 1.21.0
- Scipy >= v. 1.7.3
- Pandas >= v. 1.2.5
- Pymatgen >= v. 2022.0.9
- Matminer (for benchmarking) >= v. 0.7.3
