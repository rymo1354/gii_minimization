#@author: rymo1354

import os
from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import minimize
import time

class BVparams():

    def __init__(self, bv_file_path='bvparm16.cif'):
        ''' Taken from matminer python package '''
        parent_location = Path(os.path.abspath(__file__)).parent.absolute()
        self.bvfile = os.path.join(parent_location, bv_file_path) # Check gii_minimization directory for bv_file_path
        params = pd.read_csv(self.bvfile, sep='\s+',
                                  header=None,
                                  names=['Atom1', 'Atom1_valence',
                                         'Atom2', 'Atom2_valence',
                                         'Ro', 'B',
                                         'ref_id', 'details'],
                                  skiprows=172,
                                  skipfooter=1,
                                  index_col=False,
                                  engine="python")
        self.params = params

    def get_bv_params(self, cation, anion, cat_val, an_val):
        """Lookup bond valence parameters from IUPAC table.
        Args:
            cation (Element): cation element
            anion (Element): anion element
            cat_val (Integer): cation formal oxidation state
            an_val (Integer): anion formal oxidation state
        Returns:
            bond_val_list: dataframe of bond valence parameters
        """
        def get_params(cation, cat_oxi, anion, an_oxi):
            bond_val_list = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                    & (bv_data['Atom1_valence'] == cat_oxi) \
                                    & (bv_data['Atom2'] == str(anion)) \
                                    & (bv_data['Atom2_valence'] == an_oxi)]
            return bond_val_list.iloc[0] # Take first value if multiple exist

        bv_data = self.params
        try:
            return get_params(cation, cat_val, anion, an_val)
        except IndexError: # For single-letter cations tabulated with following _
            return get_params(cation + '_', cat_val, anion, an_val)

class GIICalculator():

    def __init__(self, params_dict=None, method='CrystalNN', **kwargs):
        ''' site_ind: (int) indice of site in Structure obj for x-coordinates to be optimized
            params_dict: (dict) Dictionary of the form:
                {'Cation' (PMG Specie Object): [],
                 'Anion' (PMG Specie Object)): [],
                 'R0' (Float): [],
                 'B' (Float): []}
            method: (str) method to identify nearest neighbors; currently supports "CrystalNN" and "Cutoff"
            **kwargs:
                cutoff: (float) cutoff radius if method='Cutoff'
                neighbor_charge: (str) charge of neighbors considered; 'opposite' or 'all' '''
        self.params_dict = params_dict
        self.method = method
        if self.method == 'Cutoff':
            self.cutoff = kwargs['cutoff']
            if 'neighbor_charge' in kwargs.keys():
                self.neighbor_charge = kwargs['neighbor_charge']
            else:
                self.neighbor_charge = 'opposite' # Default
        else:
            self.cutoff = None
            self.neighbor_charge = None

    def tab_bvparams(self, cation, anion):
        bvp = BVparams()
        val_dict = bvp.get_bv_params(str(cation.element),
                                     str(anion.element),
                                     cation.oxi_state,
                                     anion.oxi_state)
        R0 = val_dict['Ro']
        B = val_dict['B']
        return R0, B

    def get_bvparams(self, site1, site2):
        ''' site1: (pymatgen PeriodicSite obj)
            site2: (pymatgen PeriodicSite obj) '''

        def add_params(cation, anion, R0, B):
            self.params_dict['Cation'].append(cation)
            self.params_dict['Anion'].append(anion)
            self.params_dict['R0'].append(R0)
            self.params_dict['B'].append(B)
            return

        if np.sign(site1.specie.oxi_state) == 1 and np.sign(site2.specie.oxi_state) == -1:
            cation = site1.specie
            anion = site2.specie
        elif np.sign(site1.specie.oxi_state) == -1 and np.sign(site2.specie.oxi_state) == 1:
            cation = site2.specie
            anion = site1.specie
        else:
            pass # Currently only supports cation/anion pairs

        if self.params_dict != None: # If parameter dictionary passed OR has been written
            try: # Look for parameters to be passed in self.params_dict first
                params_df = pd.DataFrame.from_dict(self.params_dict)
                entry = params_df.loc[(params_df['Cation'] == cation) & (params_df['Anion'] == anion)].iloc[0]
                R0 = entry['R0']
                B = entry['B']
            except IndexError: # If Species specific parameters are not passed
                R0, B = self.tab_bvparams(cation, anion)
                add_params(cation, anion, R0, B)
        else: # Backup experimental BV parameters checked and added to self.params_dict
            self.params_dict = {'Cation': [], 'Anion': [], 'R0': [], 'B': []}
            R0, B = self.tab_bvparams(cation, anion)
            add_params(cation, anion, R0, B)

        return R0, B

    def get_neighbors(self, structure, site_ind):
        ''' Returns the neighboring sites depending on the method chosen '''
        if self.method == 'CrystalNN':
            cnn = CrystalNN(cation_anion=True, weighted_cn=True) # weighted CN so all neighbors counted
            nn_info = cnn.get_nn_info(structure, site_ind)
            neighbors = [nn_dict['site'] for nn_dict in nn_info]
        elif self.method == 'Cutoff':
            all_neighbors = structure.get_neighbors(structure[site_ind], r=self.cutoff)
            if self.neighbor_charge == 'all': # Currently not supported
                neighbors = all_neighbors
            else:
                site_oxi = structure[site_ind].specie.oxi_state
                neighbors = [s for s in all_neighbors if np.sign(s.specie.oxi_state) != np.sign(site_oxi)]
        else:
            print('Nearest neighbor method not supported')
            sys.exit(1)

        return neighbors

    def get_equivalent_sites(self, structure, symprec=0.0001, angle_tolerance=0.001):
        ''' Use symmetry operations to speed up GII calculation '''
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        sym_struct = sga.get_symmetrized_structure()
        equivs = sym_struct.equivalent_sites
        return equivs

    def sij(self, R0, B, distance):
        sij = np.exp(np.divide(np.subtract(R0, distance), B))
        return sij

    def di(self, site, neighboring_sites):
        # Note: can edit this to include same charge interactions
        bvs = 0
        for neighbor in neighboring_sites:
            distance = neighbor[1] # get distance from PeriodicSite object
            R0, B = self.get_bvparams(site, neighbor)
            sij = self.sij(R0, B, distance)
            bvs += sij

        Voxi = site.specie.oxi_state
        if np.sign(Voxi) == 1:
            return np.subtract(Voxi, bvs)
        else:
            return np.add(Voxi, bvs)

    def di_squared(self, di, weight=1):
        # Note: can change the weighting of di squared
        return np.multiply(weight, np.square(di))

    def GII(self, structure, use_sym=True):
        ''' Computes the GII of a pymatgen.core.structure.Structure object '''

        def get_site_index_di_squared(site, structure, repeats):
            site_index = structure.index(site)
            neighbors = self.get_neighbors(structure, site_index)
            di = self.di(site, neighbors)
            di_squared = self.di_squared(di)
            sym_di_squared = np.multiply(repeats, di_squared)
            return sym_di_squared

        sum_di_squared = 0
        if use_sym == True:
            try:
                equivalent_sites_list = self.get_equivalent_sites(structure) # List of equivalent sites lists
                try:
                    for site_list in equivalent_sites_list:
                        site = site_list[0]
                        sum_di_squared += get_site_index_di_squared(site, structure, len(site_list))
                except ValueError: # If symmetrization does not work
                    for site in structure:
                        sum_di_squared += get_site_index_di_squared(site, structure, 1)
            except TypeError: # Another issue with symmetrized sites
                for site in structure:
                    sum_di_squared += get_site_index_di_squared(site, structure, 1)
        else:
            for site in structure:
                sum_di_squared += get_site_index_di_squared(site, structure, 1)

        GII = np.sqrt(np.divide(sum_di_squared, len(structure)))
        return GII

class SiteClusterOptimization():

    def __init__(self, params_dict=None, method='CrystalNN', **kwargs):

        self.params_dict = params_dict
        self.method = method
        if self.method == 'Cutoff':
            self.cutoff = kwargs['cutoff']
            if 'neighbor_charge' in kwargs.keys():
                self.neighbor_charge = kwargs['neighbor_charge']
            else:
                self.neighbor_charge = 'opposite' # Default
        else:
            self.cutoff = None
            self.neighbor_charge = None

    def cluster_di_squared(self, x0, structure, site_ind):
        ''' Note: Returns the sum of discrepancy factors squared for neighbors around site to be optimized
            Note: Periodic Boundary Conditions are considered in the minimization '''

        # Replace the starting structure site with a PeriodicSite w/ x0 (1X3 Cartesian coord. array)
        minimize_s = structure.copy()
        minimize_s.replace(site_ind, species=structure[site_ind].specie,
                                      coords=x0, coords_are_cartesian=True)

        # Get the nearest neighbors to the site using the method specified in __init__
        sum_di_squared = 0
        gii_calc = GIICalculator(self.params_dict, self.method, cutoff=self.cutoff, neighbor_charge=self.neighbor_charge)
        neighbors = gii_calc.get_neighbors(minimize_s, site_ind)
        minimize_site_di = gii_calc.di(minimize_s[site_ind], neighbors)
        minimize_site_di_squared = gii_calc.di_squared(minimize_site_di)
        sum_di_squared += minimize_site_di_squared

        # Get the cluster di from the nearest neighbors
        for neighbor in neighbors:
            neighbor_neighbors = gii_calc.get_neighbors(minimize_s, neighbor.index)
            neighbor_di = gii_calc.di(neighbor, neighbor_neighbors)
            neighbor_di_squared = gii_calc.di_squared(neighbor_di)
            sum_di_squared += neighbor_di_squared

        norm_sum_di_squared = np.divide(sum_di_squared, (1 + len(neighbors))) # Normalized for # of di's computed

        return norm_sum_di_squared

    def minimize_cluster_di_squared(self, structure, site_ind, method='nelder-mead', options={'xtol': 1e-1,
                                                                                   'fatol': 1e-1,
                                                                                   'disp': False},
                                                                                    **kwargs):
        ''' Minimizes di squared for a site given its local coordination environment
            and returns a new site with coordinates that minimize di squared
            Note: Performance might decrease (unrealistic positions) for 'Cutoff' with very large cutoff '''
        result = minimize(self.cluster_di_squared, structure[site_ind].coords, args=(structure, site_ind), method=method,
                              options=options, **kwargs)

        cluster_minimized_s = structure.copy()
        cluster_minimized_s.replace(site_ind, species=structure[site_ind].specie,
                                      coords=result.x, coords_are_cartesian=True)

        return cluster_minimized_s

class GIIMinimizer():

    def __init__(self, structure, convergence_tolerance, params_dict=None, method='CrystalNN', start_val=1, **kwargs):

        self.starting_structure = structure
        self.final_structure = structure.copy()
        self.convergence_tolerance = convergence_tolerance
        self.params_dict = params_dict
        self.method = method
        if self.method == 'Cutoff':
            self.cutoff = kwargs['cutoff']
            if 'neighbor_charge' in kwargs.keys():
                self.neighbor_charge = kwargs['neighbor_charge']
            else:
                self.neighbor_charge = 'opposite' # Default
        else:
            self.cutoff = None
            self.neighbor_charge = None
        self.sco = SiteClusterOptimization(params_dict=self.params_dict, method=self.method,
                                    cutoff=self.cutoff, neighbor_charge=self.neighbor_charge)
        self.gii_calc = GIICalculator(params_dict=self.params_dict, method=self.method,
                                    cutoff=self.cutoff, neighbor_charge=self.neighbor_charge)

        self.cluster_di_squareds = self.starting_cluster_di_squared() # Compute these upon initialization
        self.diffs = [start_val for i in range(len(self.starting_structure))] # Arbitrary large value to start
        self.num_opts = [0 for i in range(len(self.starting_structure))] # Track # of optimizations performed by site

    def starting_cluster_di_squared(self):
        cluster_di_squareds = []
        for site_ind in range(len(self.starting_structure)):
            cluster_di_squared = self.sco.cluster_di_squared(self.starting_structure[site_ind].coords,
                                                            self.starting_structure, site_ind)
            cluster_di_squareds.append(cluster_di_squared)
        return cluster_di_squareds

    def choose_site(self, opt_method):
        ''' Current method to choose sites for optimization:
            1. Removes sites from consideration with abs(delGII) < convergence_tolerance
            2. Screens remaining sites by # of site optimizations performed (fewer preferred)
            3. Implements passed method to select site for optimization:
                a. max: optimizes site with the max. cluster discrepancy factor
                b. random: chooses a random site for optimization '''
        # Step 1
        site_inds_lt_tolerance = [i for i in range(len(self.diffs)) if np.abs(self.diffs[i]) > self.convergence_tolerance] # site indices that haven't converged
        least_opts_consider = [self.num_opts[i] for i in range(len(self.num_opts)) if i in site_inds_lt_tolerance] # # previous optimizations for sites t.b. considered
        # Step 2
        least_opt = np.min(least_opts_consider) # Get fewest # of optimizations performed for sites t.b. considered
        least_opts_indices = [site_inds_lt_tolerance[i] for i in range(len(site_inds_lt_tolerance)) if least_opts_consider[i] == least_opt] # site indices of fewest_opts
        fewest_opts_cluster_di_squareds = [self.cluster_di_squareds[i] for i in range(len(self.cluster_di_squareds)) if i in least_opts_indices]
        # Step 3
        if opt_method == 'max': # a
            max_cluster_di_squared = np.max(fewest_opts_cluster_di_squareds)
            cluster_site_ind = least_opts_indices[fewest_opts_cluster_di_squareds.index(max_cluster_di_squared)]
        elif opt_method == 'random': # b
            random_ind = np.random.randint(0, len(least_opts_indices))
            cluster_site_ind = least_opts_indices[random_ind]
        return cluster_site_ind

    def gii_minimization(self, opt_method='max'):
        ''' This is the main function called; optimizes the sites of the initialized structure '''
        start_total_time = time.time()
        step = 0
        while np.max(self.diffs) > self.convergence_tolerance: # Not all sites converged
            optimize_ind = self.choose_site(opt_method)
            start_GII = self.gii_calc.GII(self.final_structure, use_sym=False) # Weird behavior if use_sym=True
            self.final_structure = self.sco.minimize_cluster_di_squared(self.final_structure, optimize_ind)
            final_GII = self.gii_calc.GII(self.final_structure, use_sym=False)
            self.cluster_di_squareds[optimize_ind] = self.sco.cluster_di_squared(self.final_structure[optimize_ind].coords,
                                                            self.final_structure, optimize_ind)
            self.diffs[optimize_ind] = np.subtract(start_GII, final_GII)
            self.num_opts[optimize_ind] += 1
            step += 1
            print('Step ' + str(step) + ' complete; %s --> %s' % (str(start_GII), str(final_GII)))
        print('Convergence reached; delGII from all site optimizations < %s' % (str(self.convergence_tolerance)))
        print("Total Time: --- %s seconds ---" % (time.time() - start_total_time))

        return self.final_structure
