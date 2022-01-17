from gii_minimization.gii_calculator import BVparams, GIICalculator
from scipy.optimize import minimize, NonlinearConstraint
import sys

class ParamOuterLoop():
    def __init__(self, structures_energies, cations_anions, starting_params):
        ''' structures_energies: a dictionary object with reduced compositions as keys and the following format -
                {composition: {'structures': [list of oxidation-state decorated pymatgen.core.structure.Structure objects
                                             with the same reduced formulas, i.e. A2B4X6, ABX3, etc.],
                               'energies': [list of float objects corresponding to each structure's DFT energy]}}

            to_parameterize: a list of cation-anion pairs (tuples) to parameterize; cations and anions should be
                pymatgen.core.composition.Specie objects

            starting_params: a dictionary of starting parameters- uses same format as GIICalculator
                             (dict) Dictionary of the form:
                                 {'Cation' (PMG Specie Object): [],
                                  'Anion' (PMG Specie Object)): [],
                                  'R0' (Float): [],
                                  'B' (Float): []} '''

        self.structures_energies = structures_energies
        self.cations_anions = cations_anions
        self.starting_params = starting_params

        self.oxi_structures, self.energies, self.pairs = self.arrange_inputs()
        self.pairs_to_optimize = copy.deepcopy(self.pairs) ## Controls which pairs are optimized during each step
        self.updated_params = copy.deepcopy(self.starting_params)

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
            lengths.append(lengths)

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

    def parameter_optimization(self, lb=0.7, opt_tol=0.01, init_steps=3, max_steps=12, obj_func='gii_gs', parameterize='R0'):
        ''' lb (float): lower bound for pearson correlation constraint
            opt_tol (float): if parameters following optimization are within this tolerance, no longer optimized
            init_steps (int): opt_tol and parameter exclusion only applied after init_steps
            obj_func (str): objective function to use; currently on "gii_gs" supported
            parameterize (str): which terms to parameterize; supports "R0", "B", or "both" '''

        step = 0
        while len(self.pairs_to_optimize) > 0 or step < max_step: # still pairs left to optimize
            step += 1
            for pair in self.pairs_to_optimize:
                print(pair, step)
                opt_index = self.pairs.index(pair)

                starting_params = self.get_structure_params(self.oxi_structures[opt_index])
                param_index = self.get_cation_anion_pair_index(pair[0], pair[1], starting_params)
                starting_R0, starting_B = starting_params['R0'][param_index], starting_params['B'][param_index]
                print(starting_R0, starting_B)

                gbvpo = GeneralBVParamOptimization(self.oxi_structures[opt_index],
                                                   self.energies[opt_index],
                                                   starting_params, pair[0], pair[1], obj_func, lb, parameterize)
                final_params = gbvpo.param_optimizer()
                final_R0, final_B = final_params['R0'][param_index], final_params['B'][param_index]
                print(final_R0, final_B)
                R0_diff, B_diff = np.subtract(starting_R0, final_R0), np.subtract(starting_B, final_B)

                full_dct_param_index = self.get_cation_anion_pair_index(pair[0], pair[1], self.updated_params)
                self.updated_params['R0'][full_dct_param_index] = final_R0
                self.updated_params['B'][full_dct_param_index] = final_B

                if step >= init_steps and np.abs(R0_diff) <= opt_tol and np.abs(B_diff) <= opt_tol:
                    self.pairs_to_optimize.remove(pair)
        print('Convergence Reached')

        return

class GeneralBVParamOptimization():

    def __init__(self, oxi_structures, energies, starting_params, cation, anion, obj_func='gii_gs', lb=0.7, parameterize='R0'):

        self.oxi_structures = oxi_structures # Structures with oxidation states assigned, separated by composition
        self.energies = energies # Energies associated with structures, separated by composition
        self.starting_params = starting_params # Dictionary of starting params, format same as BVParams

        self.cation = cation # The pymatgen.core.composition.Specie cation for which to optimize R0
        self.anion = anion # The pymatgen.core.composition.Specie anion for which to optimize R0
        self.obj_function = obj_func # What to minimize- currently the ground state GII for each composition
        self.lb = lb # Pearson correlation lower bound- default is 0.7
        self.parameterize = parameterize

        self.options = {'gtol': 1e-3, 'xtol': 1e-2, 'barrier_tol': 1e-2, 'disp': True, 'verbose': 0} # Trust-Constrained Options
        self.dct_ind = self.get_cation_anion_pair_index()
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

    def get_cation_anion_pair_index(self):
        cation_inds = [i for i in range(len(self.starting_params['Cation'])) if self.cation == self.starting_params['Cation'][i]]
        anion_inds = [i for i in range(len(self.starting_params['Anion'])) if self.anion == self.starting_params['Anion'][i]]
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

        new_dict['Cation'] = self.starting_params['Cation']
        new_dict['Anion'] = self.starting_params['Anion']
        new_dict['R0'] = self.starting_params['R0']
        new_dict['B'] = self.starting_params['B']

        if self.parameterize == 'R0':
            new_dict['R0'][self.dct_ind] = x0[0] # only one parameter changing
        elif self.parameterize == 'B':
            new_dict['B'][self.dct_ind] = x0[0] # only one parameter changing
        elif self.parameterize == 'both':
            new_dict['R0'][self.dct_ind] = x0[0]
            new_dict['B'][self.dct_ind] = x0[1] # both parameters changing

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
                rg = np.subtract(np.max(se), np.min(se)) # dHd over entire dataset
                weighting = np.divide(rg, np.add(self.ground_state_energies, rg))
            for ind in range(len(gs_structures)):
                gs_gii = self.calculate_GII(gs_structures[ind], params_dict)
                sum_gs_giis += np.multiply(weighting[ind], gs_gii)
        return sum_gs_giis

    def optimize_func(self, x0):
        # x0 is the params in the following order: all R0s, followed by all Bs
        new_dict = self.get_params_dict(x0)
        if self.obj_function == 'gii_gs':
            func = self.get_sum_gs_giis(self.ground_state_structures, new_dict) # Minimize sum(GII_GS)
        else:
            sys.exit(1)
        return func

    def param_optimizer(self):
        starting_r0 = self.starting_params['R0'][self.dct_ind]
        starting_B = self.starting_params['B'][self.dct_ind]

        if self.parameterize == 'R0':
            x0 = [starting_r0]
        elif self.parameterize == 'B':
            x0 = [starting_B]
        elif self.parameterize == 'both':
            x0 = [starting_R0, starting_B]
        else:
            sys.exit(1)
        #print(x0)

        if self.obj_function == 'gii_gs':
            constraints = [NonlinearConstraint(self.pearson_constraint, self.lb, 1)] # This is used
            result = minimize(self.optimize_func, x0, method='trust-constr', options=self.options,
                               constraints=constraints)
        else:
            sys.exit(1)

        rounded_result = np.round(result.x, 3) # Rounded to be compatible with BVM convention
        #print(rounded_result)
        final_dict = self.get_params_dict(rounded_result)
        self.final_dict = final_dict

        return final_dict
