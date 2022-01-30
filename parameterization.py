#@author: rymo1354
# Date 1/17/2022

from gii_calculator import GIICalculator
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import pearsonr
from copy import deepcopy
import sys
import copy
import numpy as np

class CompositionSpecificBVParamOptimizationOuterLoop():
    def __init__(self, structures_energies, compositions, starting_params):
        ''' structures_energies: a dictionary object with reduced compositions as keys and the following format -
                {composition: {'structures': [list of oxidation-state decorated pymatgen.core.structure.Structure objects
                                             with the same reduced formulas, i.e. A2B4X6, ABX3, etc.],
                               'energies': [list of float objects corresponding to each structure's DFT energy]}}

            compositions: a list of cation-anion pairs (tuples) to parameterize; compositions can be
                strings or pymatgen.core.composition.Composition objects

            starting_params: a dictionary of starting parameters- uses same format as GIICalculator
                             (dict) Dictionary of the form:
                                 {'Cation' (PMG Specie Object): [],
                                  'Anion' (PMG Specie Object)): [],
                                  'R0' (Float): [],
                                  'B' (Float): []} '''
        self.structures_energies = structures_energies
        self.compositions = compositions
        self.starting_params = deepcopy(starting_params)

        self.oxi_structures, self.energies = self.get_composition_structures()
        self.updated_params_by_composition = {}

    def get_composition_structures(self):
        oxi_structures = []
        energies = []
        for composition in self.compositions:
            oxi_structures.append(self.structures_energies[composition]['structures'])
            energies.append(self.structures_energies[composition]['energies'])
        return oxi_structures, energies

    def get_composition_cations_anions(self, composition):
        composition_index = self.compositions.index(composition)
        structure_species = self.oxi_structures[composition_index][0].species # Get species from first entry
        unique_cations = list(np.unique([s for s in structure_species if np.sign(s.oxi_state) == 1]))
        unique_anions = list(np.unique([s for s in structure_species if np.sign(s.oxi_state) == -1]))
        cations, anions = [], []
        for uc in unique_cations:
            for ua in unique_anions:
                cations.append(uc)
                anions.append(ua)
        return cations, anions

    def parameter_optimization(self, obj_func='gii_gs', parameterize='R0', lb=0.7, use_weighting=False,
                               options={'gtol': 1e-3, 'xtol': 1e-2, 'barrier_tol': 1e-2, 'disp': True, 'verbose': 0}):
        for composition in self.compositions:
            opt_index = self.compositions.index(composition)
            cations, anions = self.get_composition_cations_anions(composition)
            bvpo = BVParamOptimization([self.oxi_structures[opt_index]], [self.energies[opt_index]],
                                               self.starting_params, cations, anions, lb=lb, use_weighting=use_weighting, obj_func=obj_func,
                                               parameterize=parameterize, options=options)
            bvpo.param_optimizer()
            self.updated_params_by_composition[composition] = bvpo.final_dict
        print('All parameters optimized using convergence criteria')

        return

class GeneralBVParamOptimizationOuterLoop():
    def __init__(self, structures_energies, cations_anions, starting_params):
        ''' structures_energies: a dictionary object with reduced compositions as keys and the following format -
                {composition: {'structures': [list of oxidation-state decorated pymatgen.core.structure.Structure objects
                                             with the same reduced formulas, i.e. A2B4X6, ABX3, etc.],
                               'energies': [list of float objects corresponding to each structure's DFT energy]}}

            cations_anions: a list of cation-anion pairs (tuples) to parameterize; cations and anions should be
                pymatgen.core.composition.Specie objects

            starting_params: a dictionary of starting parameters; MUST include all cation-anion pairs present in the dataset
                             - uses same format as GIICalculator
                             (dict) Dictionary of the form:
                                 {'Cation' (PMG Specie Object): [],
                                  'Anion' (PMG Specie Object)): [],
                                  'R0' (Float): [],
                                  'B' (Float): []} '''

        self.structures_energies = structures_energies
        self.cations_anions = cations_anions
        self.starting_params = deepcopy(starting_params)
        self.updated_params = copy.deepcopy(self.starting_params)

        self.oxi_structures, self.energies, self.pairs = self.arrange_inputs()
        self.pairs_to_optimize = copy.deepcopy(self.pairs) ## Controls which pairs are optimized during each step

    def get_cation_anion_pair_index(self, cation, anion, dct):
        cation_inds = [i for i in range(len(dct['Cation'])) if cation == dct['Cation'][i]]
        anion_inds = [i for i in range(len(dct['Anion'])) if anion == dct['Anion'][i]]
        ind = list(set(cation_inds) & set(anion_inds))[0]

        return ind

    def arrange_inputs(self):
        structures = []
        energies = []
        lengths = [] # Number of entries with cation/anion pair present

        for pair in self.cations_anions:
            pair_structures = []
            pair_energies = []
            for cmpd in list(self.structures_energies.keys()):
                check_s = self.structures_energies[cmpd]['structures'][0]
                if pair[0] in check_s.species and pair[1] in check_s.species:
                    pair_structures.append(self.structures_energies[cmpd]['structures'])
                    pair_energies.append(self.structures_energies[cmpd]['energies'])
            pair_length = len(pair_energies)
            structures.append(pair_structures)
            energies.append(pair_energies)
            lengths.append(pair_length)

        zipped = zip(lengths, self.cations_anions, structures, energies)
        sort = sorted(zipped, key = lambda t: t[0])
        s_species = [sort[i][1] for i in range(len(sort))]
        s_structures = [sort[i][2] for i in range(len(sort))]
        s_energies = [sort[i][3] for i in range(len(sort))]

        return s_structures, s_energies, s_species

    def get_structure_params(self, structures):
        cations = []
        anions = []
        for s_list in structures: # divided by same composition list
            cations += [s for s in s_list[0].species if np.sign(s.oxi_state) == 1]
            anions += [s for s in s_list[0].species if np.sign(s.oxi_state) == -1]
        unique_cations = list(np.unique(cations))
        unique_anions = list(np.unique(anions))

        unique_pairs = []
        for cat in unique_cations:
            for an in unique_anions:
                unique_pairs.append((cat, an))

        use_dict = {'Cation': [], 'Anion': [], 'R0': [], 'B': []}
        for pair in unique_pairs:
            index = self.get_cation_anion_pair_index(pair[0], pair[1], self.updated_params)
            use_dict['Cation'].append(pair[0])
            use_dict['Anion'].append(pair[1])
            use_dict['R0'].append(self.updated_params['R0'][index])
            use_dict['B'].append(self.updated_params['B'][index])

        return use_dict

    def parameter_optimization(self, obj_func='gii_gs', parameterize='R0', lb=0.7, opt_tol=0.01, init_steps=3, max_steps=12, use_weighting=False,
                               options={'gtol': 1e-3, 'xtol': 1e-2, 'barrier_tol': 1e-2, 'disp': True, 'verbose': 0}):
        ''' lb (float): lower bound for pearson correlation constraint
            opt_tol (float): if parameters following optimization are within this tolerance, no longer optimized
            init_steps (int): opt_tol and parameter exclusion only applied after init_steps
            obj_func (str): objective function to use; currently "gii_gs" and "di2_rmsd" supported
            parameterize (str): which terms to parameterize; supports "R0", "B", or "both" '''

        step = 0
        while len(self.pairs_to_optimize) > 0 and step < max_steps: # still pairs left to optimize
            step += 1
            for pair in self.pairs_to_optimize:
                print(pair, step)
                opt_index = self.pairs.index(pair)

                starting_params = self.get_structure_params(self.oxi_structures[opt_index])
                param_index = self.get_cation_anion_pair_index(pair[0], pair[1], starting_params)
                starting_R0, starting_B = starting_params['R0'][param_index], starting_params['B'][param_index]
                print(starting_R0, starting_B)

                bvpo = BVParamOptimization(self.oxi_structures[opt_index],
                                                   self.energies[opt_index],
                                                   starting_params, [pair[0]], [pair[1]], obj_func, lb, use_weighting, parameterize, options)
                final_params = bvpo.param_optimizer()
                final_R0, final_B = final_params['R0'][param_index], final_params['B'][param_index]
                print(final_R0, final_B)
                R0_diff, B_diff = np.subtract(starting_R0, final_R0), np.subtract(starting_B, final_B)

                full_dct_param_index = self.get_cation_anion_pair_index(pair[0], pair[1], self.updated_params)
                self.updated_params['R0'][full_dct_param_index] = final_R0
                self.updated_params['B'][full_dct_param_index] = final_B

                if step >= init_steps and np.abs(R0_diff) <= opt_tol and np.abs(B_diff) <= opt_tol:
                    self.pairs_to_optimize.remove(pair)
        print('All parameters optimized using convergence criteria')

        return

class BVParamOptimization():

    def __init__(self, oxi_structures, energies, starting_params, cations, anions, obj_func='gii_gs', lb=0.7, use_weighting=False, parameterize='R0',
                 options={'gtol': 1e-3, 'xtol': 1e-2, 'barrier_tol': 1e-2, 'disp': True, 'verbose': 0}):

        self.oxi_structures = oxi_structures # Structures with oxidation states assigned, separated by composition
        self.energies = energies # Energies associated with structures, separated by composition
        self.starting_params = starting_params # Dictionary of starting params, format same as BVParams

        self.cations = cations # List of pymatgen.core.composition.Specie cations for which to optimize R0
        self.anions = anions # List of pymatgen.core.composition.Specie anions for which to optimize R0; same length as cations
        # Note: di2_rmsd only takes a single cation and anion as arguments, as this parameterization is independent of other params

        self.obj_function = obj_func # What to minimize- currently the ground state GII for each composition
        self.lb = lb # Pearson correlation lower bound- default is 0.7
        self.use_weighting = use_weighting
        self.parameterize = parameterize

        self.options = options # Trust-Constrained Algorithm Options
        self.dct_inds = [self.get_cation_anion_pair_index(self.cations[i], self.anions[i],
                                                          self.starting_params) for i in range(len(self.cations))]
        self.ground_state_structures, self.ground_state_energies = self.get_gs_structures()


    def calculate_GIIs(self, structures, param_dict):
        calc_obj = GIICalculator(param_dict)
        GIIs = []
        for s in structures:
            GIIs.append(calc_obj.GII(s))
        return GIIs

    def calculate_GII(self, structure, param_dict):
        calc_obj = GIICalculator(param_dict)
        GII = calc_obj.GII(structure)
        return GII

    def calculate_di_squareds(self, structure, cation, anion, param_dict):
        # Will need to pass a cation and anion here
        calc_obj = GIICalculator(param_dict)
        specie_indices = []
        for site_ind in range(len(structure)): # get indices with species to be optimized
            if structure[site_ind].specie == cation:
                specie_indices.append(site_ind)
        di_squareds = []
        for specie_ind in specie_indices:
            neighbors = [a for a in calc_obj.get_neighbors(structure, specie_ind) if a.specie == anion]
            di = calc_obj.di(structure[specie_ind], neighbors)
            di_squared = calc_obj.di_squared(di)
            di_squareds.append(di_squared)
        return di_squareds

    def get_cation_anion_pair_index(self, cation, anion, dct):
        cation_inds = [i for i in range(len(self.starting_params['Cation'])) if cation == dct['Cation'][i]]
        anion_inds = [i for i in range(len(self.starting_params['Anion'])) if anion == dct['Anion'][i]]
        ind = list(set(cation_inds) & set(anion_inds))[0]

        return ind

    def get_gs_structures(self):
        # structures: list of lists of structures objects (sorted by composition)
        # energies: list of lists of energies objects (sorted_by_composition)
        gs_structures = []
        gs_energies = []
        for s_list_ind in range(len(self.oxi_structures)):
            gs_energy = None
            gs_ind = None
            for i in range(len(self.energies[s_list_ind])):
                if gs_energy == None or self.energies[s_list_ind][i] < gs_energy:
                    gs_energy = self.energies[s_list_ind][i]
                    gs_ind = i
            gs_structure = self.oxi_structures[s_list_ind][gs_ind]
            gs_structures.append(gs_structure)
            gs_energies.append(gs_energy)
        return gs_structures, gs_energies

    def get_params_dict(self, x0):
        new_dict = {}

        new_dict['Cation'] = deepcopy(self.starting_params['Cation'])
        new_dict['Anion'] = deepcopy(self.starting_params['Anion'])
        new_dict['R0'] = deepcopy(self.starting_params['R0'])
        new_dict['B'] = deepcopy(self.starting_params['B'])

        for i in range(len(self.dct_inds)):
            if self.parameterize == 'R0':
                new_dict['R0'][self.dct_inds[i]] = x0[i] # only one parameter changing
            elif self.parameterize == 'B':
                new_dict['B'][self.dct_inds[i]] = x0[i] # only one parameter changing
            elif self.parameterize == 'both':
                new_dict['R0'][self.dct_inds[i]] = x0[i]
                new_dict['B'][self.dct_inds[i]] = x0[i + len(self.dct_inds)] # both parameters changing

        return new_dict

    def pearson_constraint(self, x0):
        params_dict = self.get_params_dict(x0)
        pearsons = []
        for i in range(len(self.ground_state_structures)):
            pearson = self.get_pearson(self.oxi_structures[i],
                                       self.energies[i],
                                       params_dict)
            pearsons.append(pearson)
        mean_pearson = np.mean(pearsons)
        print(mean_pearson)
        return mean_pearson

    def get_pearson(self, structures, energies, param_dict):
        GIIs = self.calculate_GIIs(structures, param_dict)
        p = pearsonr(GIIs, energies)[0]
        return p

    def get_sum_gs_giis(self, gs_structures, params_dict, use_weighting=False): # Change if weighting is used here
        sum_gs_giis = 0
        if use_weighting == False:
            for ind in range(len(gs_structures)):
                gs_gii = self.calculate_GII(gs_structures[ind], params_dict)
                sum_gs_giis += gs_gii
        else:
            if len(self.ground_state_energies) == 1:
                weighting = [1]
            else:
                rg = np.subtract(np.max(self.ground_state_energies), np.min(self.ground_state_energies)) # dHd over entire dataset
                weighting = np.divide(rg, np.add(self.ground_state_energies, rg))
            for ind in range(len(gs_structures)):
                gs_gii = self.calculate_GII(gs_structures[ind], params_dict)
                sum_gs_giis += np.multiply(weighting[ind], gs_gii)
        return sum_gs_giis

    def get_sum_di_squared_rmsd(self, oxi_structures, params_dict):
        # Get the standard deviation of all bond valence values here
        all_di_squareds = [[] for i in self.cations]
        for i in range(len(self.cations)):
            for comp in oxi_structures:
                for ind in range(len(comp)):
                    di_squareds = self.calculate_di_squareds(comp[ind], self.cations[i], self.anions[i], params_dict)
                    all_di_squareds[i] += di_squareds
        minimize = np.sum([np.sqrt(np.divide(np.sum(all_di_squareds[i]), len(all_di_squareds[i]))) for i in range(len(all_di_squareds))])
        print(minimize)
        return minimize

    def optimize_func(self, x0):
        # x0 is the params in the following order: all R0s, followed by all Bs
        new_dict = self.get_params_dict(x0)
        if self.obj_function == 'gii_gs':
            func = self.get_sum_gs_giis(self.ground_state_structures, new_dict, self.use_weighting) # Minimize sum(GII_GS)
        elif self.obj_function == 'di2_rmsd':
            func = self.get_sum_di_squared_rmsd(self.oxi_structures, new_dict)
        else:
            sys.exit(1)
        return func

    def param_optimizer(self):
        starting_R0s = [self.starting_params['R0'][i] for i in self.dct_inds]
        starting_Bs = [self.starting_params['B'][i] for i in self.dct_inds]

        if self.parameterize == 'R0':
            x0 = starting_R0s
        elif self.parameterize == 'B':
            x0 = starting_Bs
        elif self.parameterize == 'both':
            x0 = starting_R0s + starting_Bs
        else:
            sys.exit(1)
        #print(x0)

        if self.obj_function == 'gii_gs':
            constraints = [NonlinearConstraint(self.pearson_constraint, self.lb, 1)] # This is used
            result = minimize(self.optimize_func, x0, method='trust-constr', options=self.options,
                               constraints=constraints)
        elif self.obj_function == 'di2_rmsd':
            result = minimize(self.optimize_func, x0, method='trust-constr', options=self.options)
        else:
            sys.exit(1)

        rounded_result = np.round(result.x, 3) # Rounded to be compatible with BVM convention
        #print(rounded_result)
        final_dict = self.get_params_dict(rounded_result)
        self.final_dict = final_dict

        return final_dict
