"""
Decomposition of a unitary NxN matrix U into a matrix product of 2x2 rotation matrices and diagonals phases. The algorithm comes from https://arxiv.org/pdf/math-ph/0103005.pdf
"""
import numpy as np


def get_decomposition(U):
    """ 
    Decompose the unitary following the algorithms given by https://arxiv.org/pdf/math-ph/0103005.pdf
    Argument:
        U (np.array) : NxN unitary matrix
    Return:
        decomposition (list of np.array): return the decomposition of the unitary

    """
    dim = len(U)
    decomposition = []

    # main loop
    for k in range(dim-1):

        # remove phase an get positives first columns
        phase = get_phase(U[:, 0])
        U = np.diag(np.exp(-1j*phase)) @ U
        J = create_rot(get_angle(U[:, 0]))

        # add the operator to the decomposition
        decomposition.append(np.diag(np.exp(1j*phase)))
        decomposition.extend(J)

        # Transform U in order to loop
        for j in J:
            U = j.T @ U

        # reduce the size of the matrix
        U = U[1:, 1:]

    phase = get_phase(U[:, 0])
    decomposition.append(np.diag(np.exp(1j*phase)))

    # cleaning of the deconposition (remove identities):
    decomposition = [
        mat for mat in decomposition if not np.allclose(mat, np.eye(len(mat)))]

    # have all the decompostion at the right size (ie NxN matrix):
    decomposition2 = []
    for mat in decomposition:
        mat2 = np.eye(dim).astype(complex)
        mat2[dim-len(mat):, dim-len(mat):] = mat
        decomposition2.append(mat2)

    return decomposition2


def reconstruct_from_decomposition(decomposition):
    """ 
    return the matrix obtained by multiplying the matrix of the decomposition
    """
    dim = len(decomposition[0])

    V = np.eye(dim).astype('complex')
    for mat in reversed(decomposition):
        V = mat @ V
    return V


def clean_decomposition(decomposition):
    return [mat for mat in decomposition if not np.allclose(mat, np.eye(len(mat)))]


def rotation(theta):
    """ return a 2 by 2 matrix rotation of angle theta"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def get_angle(vector):
    """ return the angles of the rotation to apply """
    vector = vector.real  # should already be real
    denom = 1
    c = np.zeros(len(vector)-1)

    # check if the vector is not already in the right form
    if np.allclose(vector, vector**2):
        idx = np.where(vector == 1.)[0][0]
        c[:idx] = np.pi/2
        c[idx:] = 0
        return c

    for k in range(len(vector)-1):
        c[k] = vector[k]/denom
        denom *= np.sqrt((1-c[k]**2))
    return np.arccos(c)


def get_phase(vector):
    """ return the phase of the first column"""
    return np.angle(vector)


def create_rot(list_theta):
    """ return a list of 2 by 2 rotation that preserve the first columns vectors of U but
    allow for a maximum of zero. see article """
    J_list = []
    for i, theta in enumerate(list_theta):
        J = np.eye(len(list_theta)+1).astype('complex')
        J[np.ix_([i, i+1], [i, i+1])] = rotation(theta)
        J_list.append(J)
    return J_list[::-1]


if __name__ == "__main__":

    from scipy.stats import unitary_group

    print('Generating a random unitary matrix U:')
    U = unitary_group.rvs(3)
    print(U)

    # decomposition of the matrix
    d = get_decomposition(U)

    print('The decomposition of this unitary is given by:')
    for element in d:
        print(element)

    # Now we just do a sanity check so the decomposition is right
    print('Sanity check, looking at the product of the decomposition:')
    V = reconstruct_from_decomposition(d)
    print('decomposition: {}'.format(V))

    print('U: {}'.format(U))
    print('Quadratic distance between the 2 matrices: {}'.format(
        np.sqrt(sum((abs(V-U)**2).flatten()))))
