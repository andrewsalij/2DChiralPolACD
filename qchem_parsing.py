import numpy as np
import re
import albano_params as ap
import python_util

'''
Transition enegies object for data storage.
Name is from the fact that this was previously used to parse Q-Chem (TM) .qcout files , and previously pickled objects are searching for it
'''

class TRANSITION_ENERGIES():
    '''
    Data transfer object for set of electronic transitions
    '''
    def __init__(self,excite_energies,total_energies,multiplicity,trans_moments,osc_strengths):
        '''
        Information regarding n electronic transition dipoles
        :param excite_energies: np.ndarray (n)
        :param total_energies: np.ndarray
        Total energy over time of convergence
        :param multiplicity: int
        :param trans_moments: np.ndarray (n,3)
        :param osc_strengths: np.ndarray (n)
        '''
        self.excite_energies = excite_energies.astype(np.double)
        self.total_energies = total_energies.astype(np.double)
        self.multiplicity = multiplicity
        self.trans_moments = trans_moments.astype(np.double)
        self.osc_strengths = osc_strengths.astype(np.double)
    #truncates components to allowed transitions
    def print_info(self):
        print("Energies (eV): "+str(self.excite_energies))
        print("Total Energies (eV): "+str(self.total_energies))
        print("Multiplicity: "+str(self.multiplicity))
        print("Dipole moments (a.u.):"+str(self.trans_moments))
        print("Oscillator strengths (a.u.):"+str(self.osc_strengths))
    def truncate(self,lower_bound = 0.01):
        '''truncates object to only include indices which are above some bound (default 0.01 e*a_0)'''
        indices_to_keep = np.argwhere(self.osc_strengths>lower_bound)
        self.excite_energies = self.excite_energies[indices_to_keep]
        self.total_energies = self.total_energies[indices_to_keep]
        self.trans_moments = self.trans_moments[indices_to_keep,:]
        self.osc_strengths = self.osc_strengths[indices_to_keep]

    def to_TDDFT_RESULTS(self,opt_str,tddft_str):
        return ap.TDDFT_RESULTS(self.excite_energies,self.trans_moments,self.osc_strengths,opt_str,tddft_str)

