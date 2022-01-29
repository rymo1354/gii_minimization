#@author: rymo1354
# date - 1/17/2022

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class BVparams():

    def __init__(self, bv_file_path='bvparms/bvparm2020.cif'):
        ''' Inspired by matminer BVparams '''
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

    def get_bv_params(self, cation, anion, cat_val, an_val, fix_B=True, B=0.37):
        """Lookup bond valence parameters from IUPAC table.
        Args:
            cation (Element): cation element
            anion (Element): anion element
            cat_val (Integer): cation formal oxidation state
            an_val (Integer): anion formal oxidation state
            fix_B (Boolean): whether to fix B parameter; defaults to True
            B (float): B parameter to fix GII calculation to; B=0.37 most common
        Returns:
            bond_val_list: dataframe of bond valence parameters
        """
        def get_params(cation, cat_oxi, anion, an_oxi, fix_B, B):
            if fix_B == False:
                bond_val_list = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                        & (bv_data['Atom1_valence'] == cat_oxi) \
                                        & (bv_data['Atom2'] == str(anion)) \
                                        & (bv_data['Atom2_valence'] == an_oxi)]
                return bond_val_list.iloc[0] # Take first value if multiple exist
            else:
                try:
                    bond_val_list = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                            & (bv_data['Atom1_valence'] == cat_oxi) \
                                            & (bv_data['Atom2'] == str(anion)) \
                                            & (bv_data['Atom2_valence'] == an_oxi) \
                                            & (bv_data['B'] == B)]
                    return bond_val_list.iloc[0]
                except IndexError:
                    if len(cation) == 2: # no possibility that _ is missing
                        print('%s(%s)-%s(%s) with B=%s does not exist; returning first tabulated' \
                                            % (cation, str(cat_oxi), anion, str(an_oxi), str(B)))
                    else:
                        pass
                    bond_val_list = self.params.loc[(bv_data['Atom1'] == str(cation)) \
                                            & (bv_data['Atom1_valence'] == cat_oxi) \
                                            & (bv_data['Atom2'] == str(anion)) \
                                            & (bv_data['Atom2_valence'] == an_oxi)]
                    return bond_val_list.iloc[0]

        bv_data = self.params
        try:
            return get_params(cation, cat_val, anion, an_val, fix_B, B)
        except IndexError: # For single-letter cations tabulated with following _, specific behavior of bvparm16.cif
            return get_params(cation + '_', cat_val, anion, an_val, fix_B, B)

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
            distance = neighbor.nn_distance # get distance from PeriodicSite object
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
