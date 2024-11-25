#!/usr/bin/python3
import ctypes, ctypes.util
import numpy as np
from numpy.ctypeslib import ndpointer
import warnings

import os


path_to_dll = os.path.join(os.path.dirname(__file__), "Lib/qp_solver.dll")

my_qp_solver = ctypes.cdll.LoadLibrary(path_to_dll).qp_solve
my_qp_solver.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]
my_qp_solver.restype = ctypes.c_int
is_loaded = True


def is_symmetric_positive(matrix, eps=1e-5):
    """This function checks if a given matrix is symmetric positive or not

    :param matrix:
    :param eps:
    """

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if matrix is symmetric
    if not np.array_equal(matrix, matrix.T):
        return False

    # Check if matrix is positive definite
    if not np.all(np.linalg.eigvals(matrix) >= -eps):
        return False

    return True


def qp_solver_cpp_wrapper(
    nb_cons, nb_equa_cons, nb_var, Q, p, A_all, b_all, lb, ub, X, U
):
    """This function uses the QP Solver implemented in cpp to solve the problem in python. The problem is as follows :

    minimize 0.5*x.T@Q@x + p@x under constraints Ax + B = 0 for the first nb_equa_cons rows and Ax + B >= 0 for the remaining rows
                                            and  lb <= x <= ub

    :param nb_cons: number of constraints
    :type nb_cons: int
    :param nb_equa_cons: number of equality constraints
    :type nb_equa_cons: int
    :param nb_var: number of variables
    :type nb_var: int
    :param Q: Q matrix in the problem
    :type Q: ndarray
    :param p: p matrix in the problem
    :type p: ndarray
    :param A_all: Matrix of all the constraints: first the equality constraints then the other constraints
    :type A_all: ndarray
    :param b_all: Vector of all constraints
    :type b_all: ndarray
    :param lb: Lower bound for X
    :type lb: ndarray
    :param ub: Upper bound for X
    :type ub: ndarray
    :param X: Array that will contain the results at the end of the function, of size nb_var
    :type X: ndarray
    :param U: Array that will contain the lagrange multipliers of size nb_constraints + 2*nb_var
    :type U: ndarray
    :return: an integer indicating the eventual fail of the algorithm
    """
    # We use the same notations as in the given algorithm
    mm = b_all.size
    mmax = b_all.size + lb.size + ub.size
    nmax = Q.shape[0]
    t_a = A_all.size

    # We check that dimensions are okay
    assert mm >= nb_cons
    assert mmax >= 1
    assert nmax >= nb_var
    assert t_a >= nb_var * nb_cons

    return my_qp_solver(nb_cons, nb_equa_cons, nb_var, Q, p, A_all, b_all, lb, ub, X, U)


def qp_solver_cpp(Q, p=None, G=None, h=None, A=None, b=None, lb=None, ub=None):
    """This function uses the QP Solver implemented in cpp to solve the original problem by redefining constraints as desired by the cpp function.
        It stacks the equality constraints and the inequality constraints before calling the qp_solver contained in the dll. All parameters beside Q are optional.
        The problem is as follows :

                            minimize 0.5*X.T@Q@x + p@x under constraints Ax = B and Gx <= h and  lb <= x <= ub

    :param Q: Q matrix in the problem
    :type Q: ndarray
    :param p: p matrix in the problem
    :type p: ndarray
    :param G: Matrix of inequalty constraints
    :type G: ndarray
    :param h: Vector of inequality constraints
    :type h: ndarray
    :param A: Matrix of equality constraints
    :type A: ndarray
    :param b: Vector of equality constraints
    :type b: ndarray
    :param lb: Lower bound for X
    :type lb: ndarray
    :param ub: Upper bound for X
    :type ub: ndarray

    :return: optimal vector if found
    """

    # Constants
    float_max = np.inf
    type_of_array_elements = np.float64

    # We check that the problem is initially well given, ie that matrices Q and p are fine :

    if not is_symmetric_positive(Q):
        raise ValueError("Q matrix is not symmetric")

    p_is_empty = (p is None) or (p.size == 0)
    if p_is_empty:
        warnings.warn(
            "Warning : The p matrix in the QP problem was set to 0 as it was either empty or None -- Continuing optimization !"
        )
        p = np.zeros(Q.shape[0])

    # We check which constraints matrices are empty among the ones given

    A_is_empty = (A is None) or (A.size == 0)
    G_is_empty = (G is None) or (G.size == 0)
    b_is_empty = (b is None) or (b.size == 0)
    h_is_empty = (h is None) or (h.size == 0)

    # We set some warnings in case it was not done willingly
    # If the matrices are not empty, we force them to be of type dtype = np.float64
    if A_is_empty:
        warnings.warn(
            "Warning : The given matrix of equality constraints (A) is empty -- Continuing optimization !"
        )
    else:
        if A.dtype != "float":
            A = A.astype(np.float64)
            warnings.warn(
                "Warning : The type of elements in array A was not float. Setting it to float64"
            )
        if A.ndim == 1:
            A = A.reshape(1, A.size)
            warnings.warn("Warning : A was reshaped as its shape was not of form (_,_)")
    if G_is_empty:
        warnings.warn(
            "Warning : The given matrix of inequality constraints (G) is empty -- Continuing optimization !"
        )
    else:
        if G.dtype != "float":
            G = G.astype(np.float64)
            warnings.warn(
                "Warning : The type of elements in array G was not float. Setting it to float64"
            )
        if G.ndim == 1:
            G = G.reshape(1, G.size)
            warnings.warn("Warning : G was reshaped as its shape was not of form (_,_)")
    if b_is_empty:
        warnings.warn(
            "Warning : The given vector of equality constraints (b) is empty -- Continuing optimization !"
        )
    else:
        if b.dtype != "float":
            b = b.astype(np.float64)
            warnings.warn(
                "Warning : The type of elements in array b was not float. Setting it to float64"
            )
        if b.ndim > 1:
            b = b.reshape(b.size)
            warnings.warn("Warning : b was reshaped as its shape was not of form (n,)")
    if h_is_empty:
        warnings.warn(
            "Warning : The given vector of inequality constraints (h) is empty -- Continuing optimization !"
        )
    else:
        if h.dtype != "float":
            h = h.astype(np.float64)
            warnings.warn(
                "Warning : The type of elements in array h was not float. Setting it to float64"
            )
        if h.ndim > 1:
            h = h.reshape(h.size)
            warnings.warn("Warning : h was reshaped as its shape was not of form (n,)")

    # We stack the constraints if possible
    if A_is_empty and G_is_empty:
        A_all = np.zeros(0)
        b_all = np.zeros(0)
    else:
        try:
            A_all = -np.concatenate((A, G))
        except ValueError as e:
            if A_is_empty:
                A_all = -G
            elif G_is_empty:
                A_all = -A
            else:
                raise e
        try:
            b_all = np.concatenate((b, h))
        except ValueError as e:
            if b_is_empty:
                b_all = h
            elif h_is_empty:
                b_all = b
            else:
                raise e
        # Reshape A as asked by the qpsolver function
        assert A_all.ndim == 2
        A_all = A_all.T.reshape(-1)

    # We get the parameters of the matrix
    nb_var = Q.shape[0]

    if A_is_empty and G_is_empty:
        nb_cons = 0
    else:
        nb_cons = b_all.size
    if A_is_empty:
        nb_equa_cons = 0
    else:
        nb_equa_cons = b.size

    # We define lower bounds and upper bounds if not given
    if lb is None:
        lb = -float_max * np.ones(nb_var)
    if ub is None:
        ub = float_max * np.ones(nb_var)

    # Setting the size of U vector
    mnn = nb_cons + 2 * nb_var

    # Setting X which will contain the result
    # Setting U which will contain Lagrange multipliers
    X = np.zeros(nb_var)
    U = np.zeros(mnn)

    # We process the value
    ifail_val = my_qp_solver(
        nb_cons, nb_equa_cons, nb_var, Q, p, A_all, b_all, lb, ub, X, U
    )

    # If optimal ...
    if ifail_val == 0:
        return X

    print("Valeur d'échec : " + str(ifail_val))
    # If not optimal, we raise an error
    if ifail_val == 1:
        raise StopIteration(
            "Too Many Iterations (more than 40 * (nb_var + nb_constraints))"
        )
    if ifail_val == 2:
        raise ValueError("Accuracy insufficient to satisfy convergence criterion")
    if ifail_val == 5:
        raise ValueError("The length of a working array is too short")
    if ifail_val > 10:
        raise RuntimeError("The constraints are inconsistent")

    raise Exception("Unkwown error")
    return X
