import numpy as np

'''Creates excitonic basic states combinatorically 
Andrew Salij and Roel Tempelaar'''
def construct_spin_exciton_basis(num_of_excitons, quanta_trunc=2):
    '''
    :param num_of_excitons: 
    :param quanta_trunc: int
    number upper bound (exclusive) on number of quanta of energy--that is, a value of 2
    allows for one unit of energy
    :return: 
    '''
    basis = []
    for i_minus in range(quanta_trunc):
        for i_plus in range(quanta_trunc):
            if i_plus + i_minus < quanta_trunc:
              basis.append([i_plus, i_minus, -1, -1])

    for i_minus in range(quanta_trunc):
        for i_plus in range(quanta_trunc):
            for i_exciton in range(num_of_excitons):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_plus, i_minus, i_exciton, -1])
            for i_exciton in range(num_of_excitons):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_plus, i_minus, -1, i_exciton])
    array_basis = np.array(basis)
    return array_basis

def construct_organic_spin_exciton_basis(num_excited_states,num_of_spin_excitons, quanta_trunc=2):
    '''
    Constructs polarized basis for a mixed quantum harmonic oscillator (organic) and
    spin exciton (TMD) system in order
    [gamma_plus,gamma_minus,\mu_organic,spin_up,spin_down]
    For the time being, only allows one system to be excited at a time (might change in future).
    May make more modular in the future
    :param num_of_excitons:
    :param quanta_trunc: int
    number upper bound (exclusive) on number of quanta of energy--that is, a value of 2
    allows for one unit of energy
    :return:
    '''
    basis = []
    for i_minus in range(quanta_trunc):
        for i_plus in range(quanta_trunc):
            if i_plus + i_minus < quanta_trunc:
              basis.append([i_plus, i_minus, -1,-1, -1])

    for i_minus in range(quanta_trunc):
        for i_plus in range(quanta_trunc):
            for i_exciton in range(num_of_spin_excitons):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_plus, i_minus, -1,i_exciton, -1])
            for i_exciton in range(num_of_spin_excitons):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_plus, i_minus, -1,-1, i_exciton])
    for i_plus in range(quanta_trunc):
        for i_minus in range(quanta_trunc):
            for i_exciton in range(num_excited_states):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_minus, i_plus, i_exciton,-1,-1])
    array_basis = np.array(basis)
    return array_basis

#minus, plus, flipped from standard methodology
#done to ensure that LHP (I_plus standard) is first in indexing
def construct_organic_basis(num_excited_states,quanta_trunc):
    '''
    :param num_excited_states: int
    :param quanta_trunc: int
    number upper bound (exclusive) on number of quanta of energy--that is, a value of 2
    allows for one unit of energy
    :return:
    '''
    basis = []
    for i_plus in range(quanta_trunc):
        for i_minus in range(quanta_trunc):
            if i_plus + i_minus < quanta_trunc:
              basis.append([i_minus, i_plus, -1])

    for i_plus in range(quanta_trunc):
        for i_minus in range(quanta_trunc):
            for i_exciton in range(num_excited_states):
                 if i_plus + i_minus + 1 < quanta_trunc:
                    basis.append([i_minus, i_plus, i_exciton])
    return np.asarray(basis,dtype=object)

#basis array
class BASIS():
    def __init__(self,basis_array,column_labels):
        self.basis = basis_array
        self.labels = column_labels

    def extract_subbasis(self,label):
        index = np.where(label == self.labels)[0][0]
        return self.basis[:,index]

#quanta is one more than actual number of photons
def construct_organic_vib_basis(num_excited_states,quanta_trunc,num_vib):
    '''
    Largely deprecated in favor of construct_organic_basis()
    :param num_excited_states:
    :param quanta_trunc:
    :param num_vib:
    :return:
    '''
    basis = []
    for i_plus in range(quanta_trunc):
        for i_minus in range(quanta_trunc):
            if i_plus + i_minus < quanta_trunc:
              basis.append([i_plus, i_minus, -1,-1])

    for i_plus in range(quanta_trunc):
        for i_minus in range(quanta_trunc):
            for i_exciton in range(num_excited_states):
                 if i_plus + i_minus + 1 < quanta_trunc:
                     for j in range(0,num_vib):
                        basis.append([i_plus, i_minus, i_exciton,j])
    return np.asarray(basis,dtype=object)






