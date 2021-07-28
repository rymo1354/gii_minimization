import pandas as pd
import numpy as np
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.local_env import CrystalNN
from scipy.optimize import minimize
import time

class BVparams():
    
    def __init__(self, bv_file_path='bvparm16.cif'):
        
        self.bvfile = bv_file_path
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

        bv_data = self.params
        try:
            bond_val_list = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                    & (bv_data['Atom1_valence'] == cat_val) \
                                    & (bv_data['Atom2'] == str(anion)) \
                                    & (bv_data['Atom2_valence'] == an_val)]
            return bond_val_list.iloc[0]
        except IndexError: # For single-letter cations tabulated with following _
            tab_cation = cation + '_'
            bond_val_list = self.params.loc[(bv_data['Atom1'] == str(tab_cation)) \
                                    & (bv_data['Atom1_valence'] == cat_val) \
                                    & (bv_data['Atom2'] == str(anion)) \
                                    & (bv_data['Atom2_valence'] == an_val)]
            return bond_val_list.iloc[0] # Take first value if multiple exist

class SiteClusterOptimization():
    
    def __init__(self, structure, params_dict=None, method='CrystalNN', **kwargs):
        ''' structure: (Pymatgen Structure obj) IMPORTANT: must be decorated with oxidation states
            site_ind: (int) indice of site in Structure obj for x-coordinates to be optimized
            params_dict: (dict) Dictionary of the form:
                {'Cation' (PMG Specie Object): [], 
                 'Anion' (PMG Specie Object)): [], 
                 'R0' (Float): [], 
                 'B' (Float): []}
            method: (str) method to identify nearest neighbors; currently supports "CrystalNN" and "Cutoff" 
            **kwargs: 
                cutoff: (float) cutoff radius if method='Cutoff' 
                neighbor_charge: (str) charge of neighbors considered; 'opposite' or 'all' '''
        
        self.structure = structure
        self.params_dict = params_dict
        self.method = method
        if self.method == 'Cutoff':
            self.cutoff = kwargs['cutoff']
            if 'neighbor_charge' in kwargs.keys():
                self.neighbor_charge = kwargs['neighbor_charge']
            else:
                self.neighbor_charge = 'opposite' # Default
                
    def get_bvparams(self, site1, site2):
        ''' site1: (pymatgen PeriodicSite obj) 
            site2: (pymatgen PeriodicSite obj)    
        '''
        # Determine whether sites are {cation, anion}, {anion, cation}, {anion, anion} or {cation, cation}
        # Currently only supports cation/anion pairs
        
        if np.sign(site1.specie.oxi_state) == 1 and np.sign(site2.specie.oxi_state) == -1:
            cation = site1.specie
            anion = site2.specie
        elif np.sign(site1.specie.oxi_state) == -1 and np.sign(site2.specie.oxi_state) == 1: 
            cation = site2.specie
            anion = site1.specie
        else:
            pass # Currently not supported
        
        if self.params_dict != None: #
            try: # Look for parameters to be passed in self.params_dict first
                params_df = pd.DataFrame.from_dict(self.params_dict)
                entry = params_df.loc[(params_df['Cation'] == cation) & (params_df['Anion'] == anion)].iloc[0]
                R0 = entry['R0']
                B = entry['B']
            except: # If Species specific parameters are not passed
                pass
        else: # Backup experimental BV parameters checked
            bvp = BVparams()
            val_dict = bvp.get_bv_params(str(cation.element), 
                                         str(anion.element), 
                                         cation.oxi_state, 
                                         anion.oxi_state)
            R0 = val_dict['Ro']
            B = val_dict['B']
        
        return R0, B
    
    def get_neighbors(self, structure, site_ind):
        ''' Returns the neighboring sites depending on the method chosen '''
        if self.method == 'CrystalNN':
            cnn = CrystalNN(cation_anion=True, weighted_cn=True) # Only check cation/anion distances AND use weighted CN
            nn_info = cnn.get_nn_info(structure, site_ind)
            neighbors = [nn_dict['site'] for nn_dict in nn_info]
        elif self.method == 'Cutoff':
            all_neighbors = structure.get_neighbors(structure[site_ind], r=self.cutoff)
            if self.neighbor_charge == 'all':
                neighbors = all_neighbors
            else:
                site_oxi = structure[site_ind].specie.oxi_state
                neighbors = [s for s in all_neighbors if np.sign(s.specie.oxi_state) != np.sign(site_oxi)]
        else:
            print('Nearest neighbor method not supported')
            sys.exit(1)
        
        return neighbors

    def sij(self, R0, B, distance):
        sij = np.exp(np.divide(np.subtract(R0, distance), B))
        return sij
    
    def di_squared(self, site, neighboring_sites):
        # Note: can edit this to include 1. Same type interactions or 2. Weighted discrepancy factors
        bvs = 0
        for neighbor in neighboring_sites:
            distance = neighbor[1] # get distance from PeriodicSite object
            R0, B = self.get_bvparams(site, neighbor)
            sij = self.sij(R0, B, distance)
            bvs += sij
        
        Voxi = site.specie.oxi_state
        if np.sign(Voxi) == 1:
            return np.square(Voxi - bvs)
        else:
            return np.square(Voxi + bvs)
        
    def GII(self, structure):
        sum_di_squared = 0
        for site_ind, site in enumerate(structure):
            neighbors = self.get_neighbors(structure, site_ind)
            di_squared = self.di_squared(site, neighbors)
            sum_di_squared += di_squared
        GII = np.sqrt(sum_di_squared/len(structure))
        return GII
    
    def cluster_di_squared(self, x0, site_ind):
        ''' Note: Returns the sum of discrepancy factors squared for neighbors around site to be optimized
            Note: Periodic Boundary Conditions are considered in the minimization '''
        
        sum_di_squared = 0
        
        # Replace the starting structure site with a PeriodicSite w/ x0 (1X3 Cartesian coord. array)
        minimize_s = self.structure.copy()
        minimize_s.replace(site_ind, species=self.structure[site_ind].specie, 
                                      coords=x0, coords_are_cartesian=True)
        
        # Get the nearest neighbors to the site using the method specified in __init__
        neighbors = self.get_neighbors(minimize_s, site_ind)
        minimize_site_di_squared = self.di_squared(minimize_s[site_ind], neighbors) 
        sum_di_squared += minimize_site_di_squared 
        
        # Need to get the site nearest to the nearest neighbor using create_periodic_image
        for neighbor in neighbors:
            neighbor_neighbors = self.get_neighbors(minimize_s, neighbor.index)
            neighbor_di_squared = self.di_squared(neighbor, neighbor_neighbors)
            sum_di_squared += neighbor_di_squared
            
        norm_sum_di_squared = sum_di_squared/(1 + len(neighbors)) # Normalized to # of di's computed
        
        return norm_sum_di_squared
        
    def minimize_cluster_di_squared(self, site_ind, method='nelder-mead', options={'xtol': 1e-1, 
                                                                                   'fatol': 1e-1, 
                                                                                   'disp': False}, 
                                                                                    **kwargs):
        ''' Minimizes di squared for a site given its local coordination environment
            and returns a new site with coordinates that minimize di squared'''
        result = minimize(self.cluster_di_squared, self.structure[site_ind].coords, args=(site_ind), method=method, 
                              options=options, **kwargs)
        
        cluster_minimized_s = self.structure.copy()
        cluster_minimized_s.replace(site_ind, species=self.structure[site_ind].specie, 
                                      coords=result.x, coords_are_cartesian=True)
        
        return cluster_minimized_s
    
class GIIMinimizer():
    
    def __init__(self, structure, convergence_tolerance, params_dict=None, method='CrystalNN', **kwargs):
        
        self.starting_structure = structure
        self.convergence_tolerance = convergence_tolerance
        self.params_dict = params_dict
        self.method = method
        if self.method == 'Cutoff':
            self.cutoff = kwargs['cutoff']
            if 'neighbor_charge' in kwargs.keys():
                self.neighbor_charge = kwargs['neighbor_charge']
            else:
                self.neighbor_charge = 'opposite' # Default
        
    def choose_site(self, structure, sco_obj, opt_method):
        ''' Can add different methods to choose the site for optimization; currently, site with 
            largest cluster discrepancy factor is optimized first '''
        
        if opt_method == 'max': 
            max_cluster_di = None
            max_cluster_site_ind = None
            for site_ind in range(len(structure)):
                cluster_di = sco_obj.cluster_di_squared(structure[site_ind].coords, site_ind)
                if max_cluster_di == None or max_cluster_di < cluster_di:
                    max_cluster_di = cluster_di
                    max_cluster_site_ind = site_ind
            return max_cluster_site_ind
    
    def gii_minimization(self, opt_method='max'):
        diff = 100000 # arbitrary diff (very large) to start, ensures at least one step is taken
        step = 0
        start_s = self.starting_structure
        
        if self.method == 'CrystalNN':
            sco = SiteClusterOptimization(start_s, params_dict=self.params_dict, method=self.method) 
        else: # using the r_cut method
            sco = SiteClusterOptimization(start_s, params_dict=self.params_dict, method=self.method, 
                                         cutoff=self.cutoff, neighbor_charge=self.neighbor_charge) 
        
        while diff > self.convergence_tolerance: 
            start_time = time.time()
            optimize_ind = self.choose_site(start_s, sco, opt_method)
            start_GII = sco.GII(start_s)
            final_s = sco.minimize_cluster_di_squared(optimize_ind)
            final_GII = sco.GII(final_s)
            diff = np.subtract(start_GII, final_GII)
            start_s = final_s
            step += 1 
            print('Step ' + str(step) + ' complete; %s --> %s' % (str(start_GII), str(final_GII)))
            print("Time taken: --- %s seconds ---" % (time.time() - start_time))
        print('Convergence reached; %s < %s' % (str(diff), str(self.convergence_tolerance)))  
        
        return final_s