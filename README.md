# gii-minimization.py
Minimize Global Instability Index (GII) of a structure using bond valence parameters (tabulated or user-supplied). Includes:
- GIICalculator: class to calculate the GII of an oxidation state-decorated Pymatgen structure object (pymatgen.core.structure.Structure)
- SiteClusterOptimization: class to optimize the Cartesian coordinates of a Pymatgen site (pymatgen.core.sites.PeriodicSite) to minimize GII
- GIIMinimizer: class that minimizes the GII of a Pymatgen structure object by iterative site optimization

# Dependencies
Python packages:
- Numpy
- Scipy
- Pandas
- Pymatgen
- Matminer (for benchmarking)
