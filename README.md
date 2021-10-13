# gii_minimization.py
Minimize Global Instability Index (GII) of a structure using bond valence parameters (tabulated or user-supplied). Includes:
- GIICalculator: class to calculate the GII of an oxidation state-decorated Pymatgen structure object (pymatgen.core.structure.Structure)
  - Note: This class will not work correctly for pmg v. 2020.1.10, as PeriodicNeighbor objects are returned instead of NearestNeighbor objects   
- SiteClusterOptimization: class to optimize the Cartesian coordinates of a Pymatgen site (pymatgen.core.sites.PeriodicSite) to minimize GII
- GIIMinimizer: class that minimizes the GII of a Pymatgen structure object by iterative site optimization

# Dependencies
Python packages:
- Numpy >= v. 1.21.0
- Scipy >= v. 1.6.3
- Pandas >= v. 1.2.5
- Pymatgen >= v. 2022.0.9
- Matminer (for benchmarking) >= v. 0.7.3
