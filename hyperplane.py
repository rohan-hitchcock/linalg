import numpy as np
import xxhash

import core

class Hyperplane:

    _hash_fn = xxhash.xxh64()
    _hash_rounding_precision = 5

    def __init__(self, spanning_set, intercept=None, caching=True):
        """ A class representing a subspace of R^n, or more generally a hyperplane
            which may not pass through the origin.

            Args:
                spanning_set: a set of vectors which spans the subspace. Given
                as a numpy array of shape `(m, n)`, where `n` is the dimension of
                the larger space and `m` is the number of elements in the spanning set.
                intercept: if set, specifies a translation of the span of
                `spanning_set` away from the origin.
                caching: whether this obj ect should cache certain hard-to-compute 
                values the first time it computes them, in case they are reused
        """

        # caching variables 
        self.caching = caching
        self._hash_val = None
        self._orthonormal_basis = None

        # construct the empty subspace
        if spanning_set is None:
            self._basis = None
            self._intercept = None
            return

        # construct the one-point subspace
        if len(spanning_set) == 0:
            
            if (len(spanning_set.shape) != 2 or 
                (intercept is not None and len(intercept) != spanning_set.shape[1])):
                raise ValueError("Trying to construct dimension zero subspace\
                                  but the dimension of the parent space is not clear.\
                                  `Hyperplane.create_point_space` is recommended.")            

            self._basis = spanning_set
            self._intercept = np.zeros(spanning_set.shape[1]) if intercept is None else intercept

            self._basis.flags.writeable = False
            self._intercept.flags.writeable = False

            return
        
        # check all the input data makes sense and dimensions all agree
        if len(spanning_set.shape) != 2:
            raise ValueError(f"Spanning set of shape {spanning_set.shape} is not a set of vectors.")
        
        if intercept is not None and len(intercept.shape) != 1:
            raise ValueError(f"Intercept of shape {intercept.shape} is not a single vector")

        if intercept is not None and len(intercept) != spanning_set.shape[1]:
            raise ValueError(f"The spanning set has vectors of length\
                              {spanning_set.shape[1]} but intercept is a vector of length {len(intercept)}")


        self._basis = core.get_basis(spanning_set)
        self._intercept = np.zeros(spanning_set.shape[1]) if intercept is None else intercept

        self._basis.flags.writeable = False
        self._intercept.flags.writeable = False

    def __str__(self):
        return f"{self._basis}, {self._intercept}"
    
    def __repr__(self):
        return f"Hyperplane({str(self)})"
    
    def __eq__(self, other):

        # if we happen to have cached hash values, check them first
        if (self._hash_val is not None) and (other._hash_val is not None) and (self._hash_val != other._hash_val):
            return False

        return self.is_subset(other) and other.is_subset(self)
    
    def __hash__(self):
        
        # use cached value if it is available
        if self._hash_val is not None:
            return self._hash_val

        elif self.is_empty():
            v = Hyperplane._hash_fn.intdigest()
                
        else:
            # The same hyperplane may have different representations so 
            # we cannot compute a hash using the basis and intercept directly.
            # Instead we project the standard basis for the parent space onto 
            # the hyperplane and hash the result.
            std_proj = self.projection(np.eye(self.parent_dim()))

            # due to floating point errors, if we compute a hash directly from 
            # the result we might compute different hashes for hyperplanes 
            # we would otherwise consider equal. So, we round the result to 
            # Hyperplane._hash_rounding_precision decimal places. Increasing 
            # this number may decrease hash collisions, but increasing it too
            # much might result in equal objects having different hashes
            std_proj = std_proj.round(Hyperplane._hash_rounding_precision)

            Hyperplane._hash_fn.update(std_proj)
            v = Hyperplane._hash_fn.intdigest()
            Hyperplane._hash_fn.reset()

        if self.caching:
            self._hash_val = v

        return v

    def is_empty(self):
        """ Returns True if and only if `self` is the empty vector space."""
        return self._basis is None
    
    def is_point(self):
        """ Returns True if and only if `self` is a one point space."""
        return (not self.is_empty()) and len(self._basis) == 0
    
    def is_everything(self):
        """ Returns True if and only if `self` is all of Rn, where n is the 
            dimension of the parent space of `self`.

            If `self.is_empty()` is True this method will return False.    
        """
        return (not self.is_empty()) and (self.parent_dim() == self.dim())
    
    def is_subspace(self):
        """ Returns True if and only if `self` is a vector subspace. That is, 
            if it is a hyperplane which passes through the origin
        """
        # we check both `is_zero_vector(self._intercept)` and `self.contains(np.zeros(self.dim()))`
        # since many subspaces will be represented with a zero intercept (and checking this
        # is much more efficient), but it is not guaranteed 
        return self.is_empty() or core.is_zero_vector(self._intercept) or self.contains(np.zeros(self.parent_dim()))

    def dim(self):
        """ Returns the dimension of `self`."""
        if self.is_empty():
            raise ValueError("Dimension of empty space is not defined.")
        return self._basis.shape[0]

    def parent_dim(self):
        """ Returns the dimension of the space `self` sits inside."""
        if self.is_empty():
            raise ValueError("Parent dimension of empty space is not defined.")
        return self._basis.shape[1]

    def projection(self, v):
        """ Computes the projection of a vector `v` onto `self`.
        
            In fact, `v` can be a collection of vectors, given as a numpy array 
            of any shape. In this case, the last axis (i.e. the -1 axis) is treated 
            as the coordinates of the vectors. This means that `v.shape[-1]` must equal 
            `self.parent_dim()`. The return value is a numpy array of the same 
            shape where the vectors have been projected onto the same coordinates.

            Examples:
                - If `v.shape == (3, 4)` then `v` is treated as 3 vectors in R4.
                    `self.projection(v)[i]` is the projection of `v[i]` onto self.
                - If `v.shape == (5, 3, 4)` then `v` is treated as 5 collections, 
                    each with 3 vectors from R4. E.g. 5 paths of length 3 in.
                    `self.projection(v)[i][j]` is the projection of `v[i][j]` onto self.
            
            Args:
                v: a numpy array where `v.shape[-1] == self.parent_dim()`

            Returns:
                A numpy array with the same shape as `v`, as explained above.
        """
        if self.is_empty():
            raise ValueError("Projecting onto empty vector space")
        
        if self.parent_dim() != v.shape[-1]:
            raise ValueError(f"Cannot project vectors from R{v.shape[-1]} onto a hyperplane in R{self.parent_dim()}")

        v_translated = v - self._intercept
        v_proj = np.matmul(v_translated, self.get_projection_matrix())
        return v_proj + self._intercept
    

    def get_orthonormal_basis(self):
        """ Gets an orthonormal basis for the subspace component of this hyperplane"""

        if self._orthonormal_basis is not None:
            return self._orthonormal_basis
        
        obasis = core.orthoganalize(self._basis, normalize=True)

        if self.caching:
            self._orthonormal_basis = obasis
            self._orthonormal_basis.flags.writable = False

        return obasis

    def get_projection_matrix(self):
        """ Gets the orthogonal projection matrix onto the subspace component of 
            this hyperplane
        """

        obasis = self.get_orthonormal_basis()
        return np.matmul(obasis.T, obasis)

    def distance_to(self, v):
        """ Computes the Euclidean distance from a point or set of points `v` to 
            `self`.
            
            In fact, `v` can be a collection of vectors, given as a numpy array of 
            any shape. The last axis is treated as the coordinates of the vectors. 
            See documentation for `projection` method.

            Args:
                v: a numpy array where `v.shape[-1] == self.parent_dim()`   

            Returns:
                A numpy array of shape `v.shape[:-1]` where each entry is the distance
                of the corresponding vector.
        """
        v_proj = self.projection(v)
        return np.sqrt(np.sum((v - v_proj) ** 2, axis=-1))
    
    def contains(self, v):
        """ Returns True if and only if `v` is an element of `self`.
            
            Can also be called with `v` a collection of vectors, in which case 
            True is returned if and only if all elements of `v` are contained 
            within `self` (that is, if `v` is a subset of `self`).
        """

        if self.is_empty():
            return False
        if self.is_point():
            return np.allclose(self._intercept, v)
        
        # translate everything so this space is now a true subspace
        v_trans = np.atleast_2d(v - self._intercept)

        pivot_mask = core._row_reduce_inplace(np.hstack((self._basis.T, v_trans.T)), rref=False)
        return not np.any(pivot_mask[len(self._basis):])

    def is_subset(self, other):
        """ Returns True if and only if this subspace is contained within `other`"""

        return self.is_empty() or (other.contains(self._basis) and other.contains(self._intercept - other._intercept))

    def copy(self):
        """ Returns a copy `self`"""

        if self.is_empty():
            return Hyperplane.create_empty_space()
        
        copy = Hyperplane.create_from_basis(self._basis.copy(), self._intercept.copy())
        copy.caching = self.caching
        copy._hash_val = self._hash_val

        if self._orthonormal_basis is not None:
            copy._orthonormal_basis = self._orthonormal_basis.copy()
            copy._orthonormal_basis.flags.writable = False
   
        return copy

    def intersect(self, other):
        """ Computes the intersection of `self` and `other`."""
        if self.is_empty() or other.is_empty():
            return self.create_empty_space()

        if self.parent_dim() != other.parent_dim():
            raise ValueError(f"Cannot find the intersection of a subspace of R{self.parent_dim()}\
                             and a subspace of R{other.parent_dim()}")
        
        # handle the special cases where one space is empty or a single point
        if self.is_empty() or other.is_empty():
            return Hyperplane.create_empty_space()
        if self.is_point():
            return Hyperplane.create_point_space(self._intercept.copy()) if other.contains(self._intercept) else Hyperplane.create_empty_space()
        if other.is_point():
            return Hyperplane.create_point_space(other._intercept.copy()) if self.contains(other._intercept) else Hyperplane.create_empty_space()

        # We arange things so u is the smaller of the two bases. This 
        # is not required for correctness but reduces the number of multiplications
        if self.dim() <= other.dim():
            u = self._basis
            v = other._basis
            p = self._intercept
            q = other._intercept
        else:
            u = other._basis
            v = self._basis
            p = other._intercept
            q = self._intercept
        
        # solve the system [u.T | -v.T]x = q - p, i.e. the set of all [x y].T such 
        # that u.T x + p = v.T y + q 
        mat = np.hstack((u.T, -v.T))
        b = q - p
        s, r = core.solve(mat, b)
        
        # no solution to this system means intersection is empty
        if (s is None) and (r is None):
            return Hyperplane.create_empty_space()
        
        r_u = r[:len(u)]
        s_u = s[:,:len(u)]

        # in these matrix multiplications the vectors are rows
        intercept = np.matmul(r_u, u) + p
        basis = np.matmul(s_u, u)

        # we know a priori that basis will be linearly independent
        return Hyperplane.create_from_basis(basis, intercept)
    
    # TODO: consider whether all of the methods below are necessary. 

    def get_project_to_Rdim(self):
        """ Let n = self.parent_dim() and m = self.dim(). Computes the map
            Rn ---> self ---> Rm where the first map is orthogonal projection 
            onto self (i.e. the same as `self.projection()`) and the second is 
            an isomorphism between `self` and R^m.
        """

        # TODO: this map should be possible for affine spaces too
        if not self.is_subspace():
            raise NotImplementedError()

        obasis = self.get_orthonormal_basis()
        return obasis
    
    def orthogonal_complement(self, careful=True):
        """ Computes the orthogonal complement of `self`. Only defined when `self`
            is a subspace.
            
            Args:
                careful: if `False`, skips checking whether `self` is a subspace. 
                Behaviour is not defined if `careful == False` and `self.is_subspace() == False`.
                
            Returns:
                A Hyperplane object which is the orthogonal complement of `self`.
                Inherits `caching` flag from `self`, and if this is `True` the 
                projection matrix of the return value is computed and cached.
        """

        if careful and (not self.is_subspace()):
            raise ValueError("The orthogonal complement of a non-subspace is not defined.")

        if self.is_empty():
            raise ValueError("Orthogonal complement of empty space is not defined")

        # orthogonal projection onto the orthogonal complement
        complement_proj = np.eye(self.parent_dim()) - self.get_projection_matrix()

        complement = Hyperplane(complement_proj.T, caching=self.caching)

        if self.caching:
            complement._projection_matrix = complement_proj
            complement._projection_matrix.flags.writeable = False

        return complement

    def get_kill_projection(self, careful=True):

        if careful and (not self.is_subspace()):
            raise ValueError("This is not a subspace")

        if self.is_empty():
            raise ValueError("Kill projection not defined for empty space")
                
        extended_basis = core.get_basis(np.vstack((self._basis, np.eye(self.parent_dim()))))
        orthonormal_basis = core.orthoganalize(extended_basis, normalize=True)
        complement_basis = orthonormal_basis[self.dim():]

        # TODO: remove
        assert np.allclose(self._basis, extended_basis[:self.dim()])
        assert self == Hyperplane(orthonormal_basis[:self.dim()])
        assert self.orthogonal_complement() == Hyperplane(complement_basis)
        assert core.is_orthonormal(complement_basis)

        return complement_basis
    
    def apply_transformation(self, matrix, translation=None):
        """ Applies the affine transformation f(x) = matrix @ x + translation
            to this space. 
            
            Args:
                matrix: a 2D numpy array of the appropriate shape (`matrix.shape[-1] 
                == self.parent_dim()`)
                translation: a 1D numpy array of the appropriate shape 
                (`len(translation) == matrix.shape[0]`). If `None` no translation 
                is applied.
            
            Returns:
                A Hyperplane object which is the image of `self` under this 
                transformation.
        """
        
        if translation is None:
            outdim, _ = matrix.shape
            translation = np.zeros(outdim)
        
        basis_im = self._basis @ matrix.T + translation
        intercept_im = matrix @ self._intercept + translation

        return Hyperplane(basis_im, intercept_im, caching=self.caching)


    # TODO: the naming of the methods `plane_as_function` and `line_as_function`
    # is misleading: in the first case we give one variable as a function of the 
    # other two, and in the second we give a parametric function of the line. 
    # Eventually this should be reconciled.

    def plane_as_function(self):

        if self.parent_dim() != 3:
            raise ValueError("Only defined for proper subspaces of R3")
    
        if self.dim() != 2:
            raise ValueError("Space must be a plane")

        u1, u2 = self._basis
        normal = np.cross(u1, u2)
        d = np.dot(normal, self._intercept)

        # plane satisfies the equation dot(normal, (x, y, z)) = d
        if np.isclose(normal[2], 0) and np.isclose(normal[1], 0):
            # x must be a function of the other two variables
            return ('x', lambda y, z : (d / normal[0]) * np.ones(len(y)))
        
        if np.isclose(normal[2], 0):
            # z can't be written as a function of the other two variables
            return ('y', lambda x, z : (d - normal[0] * x) / normal[1])

        return ('z', lambda x, y : (d - normal[0] * x - normal[1] * y) / normal[2])

    def line_as_function(self):

        if self.dim() != 1:
            raise ValueError("Space must be a line")
        
        # makes the parametrization exposed by this method more consistent
        b = core.unit(self._basis)
        
        return lambda t : (b.T @ np.atleast_2d(t)).T + self._intercept

    def sample(self, n, coeff_dist=None):
        """ Gets `n` random vectors from this vector space. By default, these are 
            constructed by generating coefficients for the basis vectors of `self`
            from a standard normal distribution. 
        
            Args:
                n: the number of vectors to return 
                coeff_dist: used to generate random coefficients of the basis vectors.
                When called like `coeff_dist((m, k))` it should generate a numpy array 
                of shape `(m, k)`. The entry `(i, j)` is used as the coefficient 
                of the jth basis vector in the ith vector returned.
                
            Returns:
                A numpy array of shape (n, self.dim()) where each entry is a vector 
                in this subspace.        
        """

        if self.is_empty():
            raise ValueError("Cannot sample from an empty vector space")
        
        # return n copies of the point
        if self.is_point():
            return np.tile(self._intercept, (n, 1))

        if coeff_dist is None:
            rng = np.random.default_rng()
            coeff_dist = rng.standard_normal

        coeffs = coeff_dist((n, self.dim()))

        return np.matmul(coeffs, self._basis) + self._intercept

    @classmethod
    def create_point_space(point):
        """ Creates a Hyperplane object consisting only of `point`."""

        parent_dim = len(point)
        # an empty array of vectors with parent_dim entries
        basis = np.zeros(shape=(0, parent_dim), dtype=float)
        return Hyperplane(basis, point)

    @classmethod
    def create_zero_space(parent_dim):
        """ Creates a Hyperplane object consisting of the zero vector in the given
            dimension
        """
        # an empty array of vectors with parent_dim entries
        basis = np.zeros(shape=(0, parent_dim), dtype=float)
    
        return Hyperplane(basis)

    @classmethod
    def create_from_basis(basis, intercept=None):
        """ Creates a subspace directly from a basis. Performs no 
            checking on input data. If `basis` is not a
            basis and `intercept` (if given) not a vector in the same dimension
            then the behavior of this object is undefined.    
        """    
    
        out = Hyperplane.create_empty_space()

        if intercept is None:
            intercept = np.zeros(basis.shape[1])

        out._basis = basis
        out._intercept = intercept
        
        out._basis.flags.writeable = False
        out._intercept.flags.writeable = False

        return out

    @classmethod
    def create_empty_space():
        return Hyperplane(None)
    
    @classmethod
    def create_solution_space(a, b=None):
        return Hyperplane(*core.solve(a, b))
    