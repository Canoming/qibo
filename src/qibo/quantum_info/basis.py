from functools import reduce
from itertools import product

import numpy as np

from qibo import matrices
from qibo.config import raise_error


def vectorization(state, order: str = "row"):
    """Returns state :math:`\\rho` in its Liouville
    representation :math:`\\ket{\\rho}`.

    Args:
        state: state vector or density matrix.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.

    Returns:
        Liouville representation of ``state``.
    """

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"Object must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if (order != "row") and (order != "column") and (order != "system"):
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    if len(state.shape) == 1:
        state = np.outer(state, np.conj(state))

    if order == "row":
        state = np.reshape(state, (1, -1), order="C")[0]
    elif order == "column":
        state = np.reshape(state, (1, -1), order="F")[0]
    else:

        # def _block_split(matrix, nrows: int, ncols: int):
        #     """Block-vectorization of a square :math:`N \times N`
        #     matrix into 4 :math:`\frac{N}{2} \times \frac{N}{2}`
        #     matrices, where :math:`N = 2^{n}` and :math:`n` is the
        #     number of qubits.

        #     Args:
        #         matrix: :math:`N \times N` matrix.
        #         nrows (int): number of rows of the block matrix.
        #         ncols (int): number of columns of the block matrix

        #     Returns:
        #         Block-vectorization of ``matrix``.
        #     """
        #     dim, _ = matrix.shape
        #     return (
        #         matrix.reshape(int(dim / nrows), nrows, -1, ncols)
        #         .swapaxes(1, 2)
        #         .reshape(-1, nrows, ncols)[[0, 2, 1, 3]]
        #     )

        d = len(state)
        # n = int(d / 2)
        nqubits = int(np.log2(d))

        # if n == 1:
        #     state = state.reshape((1, -1), order="F")[0]
        # else:
        #     state = _block_split(state, n, n)
        #     for _ in range(nqubits - 2, 0, -1):
        #         n = int(n / 2)
        #         state = np.array([_block_split(matrix, n, n) for matrix in state])
        #         state = state.reshape((np.prod(state.shape[:-2]), *(state.shape[-2:])))
        #     state = np.array(
        #         [matrix.reshape((1, -1), order="F") for matrix in state]
        #     ).flatten()

        new_axis = []
        for x in range(nqubits):
            new_axis += [x + nqubits, x]
        state = np.reshape(
            np.transpose(np.reshape(state, [2] * 2 * nqubits), axes=new_axis), -1
        )

    return state


def unvectorization(state, order: str = "row"):
    """Returns state :math:`\\rho` from its Liouville
    representation :math:`\\ket{\\rho}`.

    Args:
        state: :func:`vectorization` of a quantum state.
        order (str, optional): If ``"row"``, unvectorization is performed
            row-wise. If ``"column"``, unvectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Default is ``"row"``.

    Returns:
        Density matrix of ``state``.
    """

    if len(state.shape) != 1:
        raise_error(
            TypeError,
            f"Object must have dims (k,), but have dims {state.shape}.",
        )

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if (order != "row") and (order != "column") and (order != "system"):
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    d = int(np.sqrt(len(state)))

    if (order == "row") or (order == "column"):
        order = "C" if order == "row" else "F"
        state = np.reshape(state, (d, d), order=order)
    else:
        nqubits = int(np.log2(d))
        axes_old = list(np.arange(0, 2 * nqubits))
        state = np.reshape(
            np.transpose(
                np.reshape(state, [2] * 2 * nqubits),
                axes=axes_old[1::2] + axes_old[0::2],
            ),
            [2**nqubits] * 2,
        )

    return state


def pauli_basis(nqubits: int, normalize: bool = False, vectorize: bool = False):
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis is returned.
            Defaults to False.
        vectorize (bool, optional): If ``False``, returns a nested array with
            all Pauli matrices. If ``True``, retuns an array where every
            row is a vectorized Pauli matrix. Defaults to ``False``.

    Returns:
        list: list with all Pauli matrices forming the basis.
    """

    if nqubits <= 0:
        raise_error(ValueError, "nqubits must be a positive int.")

    if not isinstance(normalize, bool):
        raise_error(
            TypeError,
            f"normalize must be type bool, but it is type {type(normalize)} instead.",
        )

    if not isinstance(vectorize, bool):
        raise_error(
            TypeError,
            f"vectorize must be type bool, but it is type {type(vectorize)} instead.",
        )

    basis = [matrices.I, matrices.X, matrices.Y, matrices.Z]

    if vectorize:
        basis = [matrix.reshape((1, -1), order="F")[0] for matrix in basis]

    if nqubits >= 2:
        basis = list(product(basis, repeat=nqubits))
        if vectorize:
            basis = [reduce(np.outer, matrix).ravel() for matrix in basis]
        else:
            basis = [reduce(np.kron, matrix) for matrix in basis]

    basis = np.array(basis)

    if normalize:
        basis /= np.sqrt(2**nqubits)

    return basis


def comp_basis_to_pauli(nqubits: int, normalize: bool = False):
    """Unitary matrix :math:`U` that converts operators from the Liouville
    representation in the computational basis to the Pauli-Liouville
    representation.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, \\ketbra{k}{P_{k}} \\,\\, ,

    where :math:`\\ket{P_{k}}` is the system-vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`\\ket{k}` is the computational
    basis element.

    When converting a state :math:`\\ket{\\rho}` to its Pauli-Liouville
    representation :math:`\\ket{\\rho'}`, one should use ``order="system"``
    in :func:`vectorization`.

    Example:
        .. code-block:: python

            from qibo.quantum_info import random_density_matrix, vectorization, comp_basis_to_pauli
            nqubits = 2
            d = 2**nqubits
            rho = random_density_matrix(d)
            U_c2p = comp_basis_to_pauli(nqubits)
            rho_liouville = vectorization(rho, order="system")
            rho_pauli_liouville = U_c2p @ rho_liouville

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            Pauli basis. Defaults to ``False``.

    Returns:
        Unitary matrix :math:`U`.

    """

    unitary = pauli_basis(nqubits, normalize, vectorize=True)
    unitary = np.conj(unitary)

    return unitary


def pauli_to_comp_basis(nqubits: int, normalize: bool = False):
    """Unitary matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, \\ketbra{P_{k}}{b_{k}} \\, .

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            Pauli basis. Defaults to ``False``.

    Returns:
        Unitary matrix :math:`U`.
    """

    matrix = np.transpose(np.conj(comp_basis_to_pauli(nqubits, normalize)))

    return matrix
