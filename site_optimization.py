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
