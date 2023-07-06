import numpy as np

def is_zero_vector(v):
    """ Checks whether a vector (or a collection of vectors) is the zero vector,
        allowing for numerical errors.

        Args:
            v: A 1D or 2D numpy array. In the 2D case we check whether each `v[i]`
            a zero vector.

        Returns:
            A boolean value (when `v` is 1D) or an array of boolean values (when
            `v` is 2D). In the 2D case, the ith entry of the return value is `True`
            if and only if `v[i]` is the zero vector.
    """
    return np.all(np.isclose(v, 0), axis=-1)


def unit(v):
    """ Computes the unit vector of a given vector or collection of vectors.

        Args:
            v: A 1D or 2D numpy array. In the 2D case we find the unit vector of
            each `v[i]`

        Returns:
            A numpy array of the same shape as `v`.

        Raises:
            ValueError: raised if `v` or any `v[i]` is the zero vector
    """
    if np.any(is_zero_vector(v)):
        raise ValueError("Finding the unit vector of a zero vector.")

    norm = np.linalg.norm(v, axis=-1)
    norm = np.atleast_2d(norm).T

    # match input shape to output shape
    return np.squeeze(v / norm) if len(v.shape) == 1 else v / norm


def projection(v, u, projecting_onto_unit=False):
    """ Computes the projection of a vector or collection of vectors onto a
        given vector.

        Args:
            v: A 1D or 2D numpy array. In the 2D case we find the projection of
            each `v[i]`
            u: A 1D numpy array. The vector we project onto.
            projecting_onto_unit: Can be set to `True` if `u` is known to be a unit
            vector to avoid unnecessary calculations (for example, to avoid unnecessarily
            accumulating numerical errors).

        Returns:
            A numpy array of the same shape as `v` consisting of projections onto `u`.

        Raises:
            ValueError: if `u` is the zero vector.
    """
    if is_zero_vector(u):
        raise ValueError("Projecting onto zero vector.")

    u = np.atleast_2d(u)
    v = np.atleast_2d(v)

    # computes the projection matrix. Note that `u` and each entry of `v` is a row vector
    projection_matrix = np.matmul(u.T, u)
    if not projecting_onto_unit:
        projection_matrix = projection_matrix / np.sum(u ** 2)

    return np.matmul(v, projection_matrix)


def orthoganalize(basis, normalize=False, copy=True):
    """ Given a basis, computes an orthogonal basis for the same space.

        Uses the modified Gram-Schmidt algorithm, which has better numerical
        properties than the classical Gram-Schmidt algorithm. See Å. Björck (1994)
        “Numerics of Gram-Schmidt orthogonalization” doi: 10.1016/0024-3795(94)90493-6.

        Args:
            basis: A 2D numpy array where each entry is an element of the basis.
            Must be linearly independent.
            normalize: If set to `True`, computes an orthonormal basis.
            copy: If set to `False`, tries to perform the algorithm in-place 
            and overwrites basis. If the entries of `basis` are not of type `float`
            then a copy will be performed anyway.

        Returns:
            A numpy array of the same shape as `basis` which is an orthonormal basis
            for the same space as `basis`.
    """

    if copy or basis.dtype != float:
        orthogonal_basis = basis.astype(float)
    else:
        orthogonal_basis = basis

    if len(orthogonal_basis) == 0:
        return orthogonal_basis

    if normalize:
        orthogonal_basis[0] = unit(orthogonal_basis[0])

    for i in range(1, len(basis)):
        orthogonal_basis[i:] -= projection(orthogonal_basis[i:], orthogonal_basis[i-1], projecting_onto_unit=normalize)
        if normalize:
            orthogonal_basis[i] = unit(orthogonal_basis[i])

    return orthogonal_basis


def is_orthonormal(vectors):
    """ Given a collection of vectors, returns True if and only if they are
        orthonormal.
    """
    dot_products = np.matmul(vectors, vectors.T)
    return np.allclose(dot_products, np.eye(len(vectors)))


def row_reduce(matrix, rref=True):
    """ Performs Gaussian elimination to row reduce a matrix.

        Args:
            matrix: A matrix as a 2D numpy array.
            rref: if `True`, computes the reduced row echelon form of `matrix`.
            Otherwise, only a row echelon form is computed.

        Returns:
            A tuple of the form `(mat, pmask)` where:
                - `mat` is either the reduced row echelon form (if `rref=True`) or a row echelon
                form of `matrix`, as a 2D numpy array.
                - `pmask` is a boolean numpy array of length equal to the
                number of columns of `mat`. An entry is `True` if and only if the
                corresponding column of `mat` is a pivot column.
    """
    matrix = matrix.copy().astype(float)
    return matrix, _row_reduce_inplace(matrix, rref=rref)


def _row_reduce_inplace(matrix, rref=True):
    """ The same as `row_reduce`, but the algorithm is performed in-place.

        Another important difference to `row_reduce` is that this function assumes
        `matrix.dtype == float`.

        Args:
            matrix: A matrix as a 2D numpy array with `matrix.dtype == float`.
            rref: if `True`, computes the reduced row echelon form of `matrix`.
            Otherwise, only a row echelon form is computed.

        Returns:
            A boolean numpy array of length equal to the number of columns of
            `matrix`. An entry is `True` if and only if the corresponding column
            in the row echelon form of `matrix` is a pivot column.

            Also modifies `matrix` in-place.
    """
    # need a matrix, not another type of numpy array
    assert len(matrix.shape) == 2

    num_columns = len(matrix.T)
    pivot_mask = np.zeros(num_columns, dtype=bool)
    num_pivots = 0

    for column in range(num_columns):

        # get the entries in this column below the last pivot
        entries = matrix.T[column, num_pivots:]

        # get the indices of the non-zero entries below the last pivot in this column
        non_zero_entries = np.flatnonzero(np.logical_not(np.isclose(entries, 0))) + num_pivots

        # this is not a pivot column
        if len(non_zero_entries) == 0:
            continue

        # make the first non-zero entry equal to one
        matrix[non_zero_entries[0], num_pivots:] /= matrix[non_zero_entries[0], column]

        # make the other non-zero entries equal to zero
        for nze in non_zero_entries[1:]:
            matrix[nze, num_pivots:] -= matrix[nze, column] * matrix[non_zero_entries[0], num_pivots:]

        # put the pivot as high as possible in this column
        row_swap(matrix, non_zero_entries[0], num_pivots)

        # mark this column as a pivot column
        pivot_mask[column] = True
        num_pivots += 1

    # if we just want row echelon form we can return now
    if not rref:
        return pivot_mask

    for pivot_row, pivot_column in enumerate(np.flatnonzero(pivot_mask)):
        for row in range(pivot_row):
            if not np.isclose(matrix[row, pivot_column], 0):
                matrix[row, pivot_column:] -= matrix[row, pivot_column] * matrix[pivot_row, pivot_column:]

    return pivot_mask


def row_swap(matrix, i, j):
    """ Swaps rows `i` and `j` in a 2D numpy array `matrix`. Modifies `matrix` in-place"""
    tmp = matrix[i].copy()
    matrix[i] = matrix[j]
    matrix[j] = tmp


def get_basis(vectors):
    """ Given a collection of vectors, returns a basis for their span."""
    _, pivot_mask = row_reduce(vectors.T, rref=False)
    return vectors[pivot_mask]


def rank(matrix):
    """ Computes the rank of `matrix`."""
    _, pivot_mask = row_reduce(matrix, rref=False)
    return np.count_nonzero(pivot_mask)


def is_linearly_independent(vectors):
    """ Returns True if and only if `vectors` is linearly independent."""
    _, pivot_mask = row_reduce(vectors.T, rref=False)
    return np.all(pivot_mask)


def solve(a, b=None):
    """ Solves a system of linear equations of the form a * x = b. 

        The return values of this function are compatible with the `Hyperplane` class.
        So, `Hyperplane(*solve(a, b))` will produce a subspace which represents 
        the set of all solutions to this equation.
    
        Args:
            a: A matrix as a numpy array.
            b: A vector as a 1D numpy array. If None, we assume `b=0`
            
        Returns:
            A tuple of the form `(vecs, intercept)`. All solutions are of the form 
            `sum(t * v for zip(ts, vecs)) + intercept`, for any vector of coefficients 
            `ts`. 

            Note that if `vecs` is empty then this system has the unique solution 
            `intercept`.

            A return value of `(None, None)` indicates the system has no solution.
    """
    assert len(a.shape) == 2
    
    # dimension of the solution vectors is equal to the number of columns
    _, vec_dim = a.shape

    # inhomogeneous system of equations
    if b is not None:
        
        # row reduce the augmented matrix [a | b]
        b = np.atleast_2d(b).T
        rref, pivot_mask = row_reduce(np.hstack((a, b)))

        # the system has no solutions if the augmented column has a pivot
        if pivot_mask[-1]:
            return (None, None)
        
        # indices of the pivot columns and non-pivot columns, 
        # excluding the augmented column
        pivot_index = np.flatnonzero(pivot_mask[:-1])
        non_pivot_index = np.flatnonzero(np.logical_not(pivot_mask[:-1]))
        
        # compute the constant vector in the vector equation of the solution using 
        # the augmented column of the rref
        intercept = np.zeros(vec_dim)
        for j, pi in enumerate(pivot_index):
            intercept[pi] = rref.T[-1][j]

        # drop the augmented column as we have finished processing it
        rref = rref[:,:-1]
    
    # homogeneous system of equations
    else:
        # as above, but there is no intercept vector to compute and we don't bother 
        # adding an augmented column of zeros to the matrix
        rref, pivot_mask = row_reduce(a)
        intercept = np.zeros(vec_dim)

        pivot_index = np.flatnonzero(pivot_mask)
        non_pivot_index = np.flatnonzero(np.logical_not(pivot_mask))

    
    # we now compute the vectors which span the set of solutions

    basis = np.zeros((len(non_pivot_index), vec_dim))

    # each non-pivot column corresponds to exactly one basis vector
    for i, npi in enumerate(non_pivot_index):
        
        # the npi-th variable is a free variable, so we put a 1 in the npi-th 
        # entry of the basis vector generated by this pivot column
        basis[i][npi] = 1

        # for pivot variables, the corresponding entry in this basis vector is 
        # determined by the entry in the same row as the pivot in the non-pivot 
        # column generating this basis vector 
        for j, pi in enumerate(pivot_index):
            basis[i][pi] = -rref.T[npi][j]

    return (basis, intercept)
