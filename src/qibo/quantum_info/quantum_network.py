#%%
import numpy as np
import re

from typing import Tuple, Union, Optional

from qibo.gates.channels import UnitaryChannel,DepolarizingChannel
from qibo.quantum_info import random_density_matrix, random_unitary
from qibo.backends import NumpyBackend

class QuantumNetwork:
    """
    Choi operator of Quantum Networks.
    ---
    The Choi operator of a quantum channel is a tensor that represents the action of the channel.

    If the operator is rank-1, it may be saved as a *pure* Choi operator. In the qubit case, a pure Choi operator is equivalent to a unitary operator or a pure state.
    **Qeustion**: if the channel is pure, should we save it as a pure Choi operator? It does save some memory, but it may be confusing and hard to use. In practice, we need to convert it to a full Choi operator for most operations, such as addition. That will cost a lot of computational overhead.
    """

    def __init__(self, mat:np.ndarray, partition:Tuple[int], sys_out:Tuple[bool]=None, is_pure:bool=False):
        """
        Given a matrix and a partition, create a Choi operator.

        arguments:
        ---
        mat: np.ndarray
            The input matrix.
        partition: Tuple[int]
            The partition of the input matrix.
        sys_out: Tuple[bool]
            The mask on the output system of the Choi operator.
        is_pure: bool
            If the input matrix is a pure Choi operator.
        """

        self._is_pure = is_pure
        self.partition = partition
        dim:int = 1
        for par in self.partition:
            dim *= par
        self.dim:int = dim
        self.set_tensor(mat, partition, sys_out, is_pure)

        if self.is_hermitian is False:
            raise Warning("The input matrix is not Hermitian.")

    def set_tensor(self,mat,partition:Tuple[int],sys_out:Tuple[bool],is_pure:bool) -> None:
        """
        Check if the input is valid.
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError("Input matrix must be a numpy array.")

        if is_pure:
            self.mat:np.ndarray = mat.reshape(partition)
        else:
            mat_partition = partition+partition
            self.mat:np.ndarray = mat.reshape(tuple(mat_partition))

        if sys_out is None:
            if len(partition) == 1:
                # If there is only one system, we assume its a quantum state.
                self.sys_out = (False,)
            if len(partition) == 2:
                # If there are two systems, we assume its a quantum channel.
                # The output system is assumed to be the second one.
                self.sys_out = (False,True)
        else:
            if len(sys_out) != len(partition):
                raise ValueError("The length of `sys_out` must be the same as the length of `partition`.")
            self.sys_out = tuple(sys_out)

    @property
    def is_pure(self) -> bool:
        return self._is_pure

    @property
    def is_hermitian(self) -> bool:
        """
        Check if the Choi operator is Hermitian.
        """
        if self.is_pure:
            self.full
        
        mat = self.mat.reshape((self.dim,self.dim))
        
        return np.allclose(
            mat, np.einsum('ij -> ji',mat.conj()))
    
    @property
    def is_sdp(self) -> bool:
        """
        Check if the Choi operator is positive semi-definite.
        """
        if self.is_pure:
            self.full
        
        mat = self.mat.reshape((self.dim,self.dim))

        return np.all(np.linalg.eigvalsh(mat) >= -1e-8)
    
    @property
    def is_unital(self) -> bool:
        """
        Check if the Choi operator is unital.
        This may not be true for general channels.
        """
        if self.is_pure:
            self.full
        
        Iin = np.eye(self.mat.shape[1])
        traced = np.einsum('ijil ->jl', self.mat)
        return np.allclose(
            traced, Iin)

    @property
    def is_causal(self) -> bool:
        """
        Check if the Choi operator satisfies the causal order condition.
        """
        if self.is_pure:
            self.full
        
        Iout = np.eye(self.mat.shape[1])
        traced = np.einsum('ijkj ->ik', self.mat)
        return np.allclose(
            traced, Iout)

    @property
    def is_channel(self) -> bool:
        """
        Check if the Choi operator is a channel.
        """
        return self.is_sdp and self.is_causal

    @property
    def full(self) -> np.ndarray:
        if self.is_pure:
            self.mat = np.einsum('ij,kl->jilk', self.mat, self.mat.conj())
            self._is_pure = False
            return self.mat
        else:
            return self.mat

    def apply(self, state:np.ndarray) -> np.ndarray:
        """
        Apply the Choi operator to the state. *Assuming the state is a density matrix*.
        This method is used to numpy arrrays.
        """
        if self.is_pure:
            return np.einsum('ij,kl,ik->jl', self.mat, self.mat.conj(), state)
        else:
            return np.einsum('ijkl,jl->ik', self.mat, state)

    def __add__(self, other:'QuantumNetwork') -> 'QuantumNetwork':
        """
        Add two Choi operators.
        The summation always return a non-pure Choi operator.
        """
        if not isinstance(other, QuantumNetwork):
            raise TypeError("Input must be a Choi operator.")
        if self.is_pure:
            this_mat = self.full
        else:
            this_mat = self.mat
        if other.is_pure:
            other_mat = other.full
        else:
            other_mat = other.mat
        if this_mat.shape != other_mat.shape:
            raise ValueError("The input Choi operators must have the same shape.")
        if self.sys_out != other.sys_out:
            raise ValueError("The input Choi operators must have the same output system.")

        return QuantumNetwork(self.mat + other.mat, self.partition, self.sys_out)
    
    def __truediv__(self, other:Union[int,float]) -> 'QuantumNetwork':
        """
        Divide a Choi operator by a scalar.
        """
        if not isinstance(other, (int,float)):
            raise TypeError("Input must be a scalar.")
        if self.is_pure:
            this_mat = self.full
        else:
            this_mat = self.mat

        return QuantumNetwork(this_mat / other, self.partition, self.sys_out)
    
    def __mul__(self, other:Union[int,float]) -> 'QuantumNetwork':
        """
        Multiply a Choi operator by a scalar.
        """
        if not isinstance(other, (int,float)):
            raise TypeError("Input must be a scalar.")
        if self.is_pure:
            this_mat = self.full
        else:
            this_mat = self.mat

        return QuantumNetwork(this_mat * other, self.partition, self.sys_out)
    
    @staticmethod
    def _channle_expr(expr):
        return re.match(
            r'i\s*j\s*,\s*j\s*k\s*->\s*i\s*k',
            expr)
    @staticmethod
    def _inv_expr(expr):
        return re.match(
            r'i\s*j\s*,\s*k\s*i\s*->\s*k\s*j',
            expr)

    def link(self, other:'QuantumNetwork', expr:str=None) -> 'QuantumNetwork':
        """
        The link product of two Choi operators.
        Note that the link product is not commutative.
        Here we assume A.link(B) means *apply B to A*.
        However, link product is associative, and we override `@` for simplify the notations.
        """
        if not isinstance(other, QuantumNetwork):
            raise TypeError("Input must be a Choi operator.")
        
        if self.is_pure:
            this_mat = self.full
        else:
            this_mat = self.mat
        if other.is_pure:
            other_mat = other.full
        else:
            other_mat = other.mat

        if expr is None or self._channle_expr(expr) is not None:
            cexpr = 'ijab,jkbc->ikac'
            return QuantumNetwork(
                np.einsum(cexpr, this_mat, other_mat),
                [self.partition[0],other.partition[1]])
        elif self._inv_expr(expr) is not None:
            cexpr = 'ijab,jkbc->ikac'
            return QuantumNetwork(
                np.einsum(cexpr, other_mat, this_mat),
                [other.partition[0],self.partition[1]])
    
    def __matmul__(self,B:'QuantumNetwork') -> 'QuantumNetwork':
        if len(self.partition) == 2 and len(B.partition) == 2:
            return self.link(B)
    
    def __str__(self) -> str:
        ind_in = [str(self.partition[i]) for i in range(len(self.partition)) if self.sys_out[i] is False]
        str_in = ", ".join(ind_in)
        ind_out = [str(self.partition[i]) for i in range(len(self.partition)) if self.sys_out[i] is True]
        str_out = ", ".join(ind_out)
        return f'J[{str_in} -> {str_out}]'

# %%
if __name__ == "__main__":
    backend = NumpyBackend()

    # The `DepolarizingChannel` is a subclass of `UnitaryChannel`.
    # But the name of `UnitaryChannel` is misleading, it defines **mixed unitary channels**.
    # The name *unitary channel* is used in the sense that the channel is defined by a single unitary matrix.
    test_ch = DepolarizingChannel(0,0.5)
    
    # The matrix method should return the Choi matrix, but it does not.
    try:
        test_ch.matrix()
    except AttributeError as err:
        print(f'The `matrix` method of a channel should return the Choi matrix, but it prompts the error: {err}.')

    # The tensor structure of the Choi matrix
    # For a quantum Channle, the Choi matrix is a 4-tensor of the shape $(n,m,n,m)$, where $n$ is the dimension of the input system, and $m$ is the dimension of the output system.
    # In qubit systems, they are assuemed to be the same here.
    N = len(test_ch.target_qubits)
    # The repeatition of the indices is included in `ChoiOp`
    # We only specify the size of input and output here.
    partition = (2**N, 2**N)
    depolar_choi = QuantumNetwork(test_ch.to_choi(), partition)
    assert depolar_choi.mat.shape == (2,2,2,2)
    assert depolar_choi.dim == 4
    assert depolar_choi.partition == (2,2)
    assert depolar_choi.sys_out == (False, True)

    assert depolar_choi.is_causal
    assert depolar_choi.is_unital
    assert depolar_choi.is_hermitian
    assert depolar_choi.is_sdp

    # Test state
    # state = np.array([[0.5,0.4],[0.4,0.5]])
    state = random_density_matrix(2)
    state_choi = QuantumNetwork(state, (1,2))

    assert state_choi.is_hermitian
    assert state_choi.is_sdp

    state_out = test_ch.apply_density_matrix(backend, state, 1)
    # Test `apply`` method
    choi_op = depolar_choi.apply(state)
    assert np.allclose(choi_op, state_out)
    # Test `link` method
    assert np.allclose(state_choi.link(depolar_choi).mat.flatten(), state_out.flatten())
    # Test the ``@`` operator on states
    assert np.allclose((state_choi @ depolar_choi).mat.flatten(), state_out.flatten())

    U = random_unitary(4)
    V = random_unitary(4)

    ChoiU = QuantumNetwork(U, (4,4), is_pure=True)
    ChoiV = QuantumNetwork(V, (4,4), is_pure=True)

    ChoiUV = QuantumNetwork(V@U, (4,4), is_pure=True)

    assert ChoiU.is_hermitian
    assert ChoiU.is_causal
    assert ChoiU.is_unital
    assert ChoiU.is_sdp

    # Test `link` method for channels
    assert np.allclose(ChoiU.link(ChoiV).mat, ChoiUV.full)
    assert np.allclose(ChoiU.link(ChoiV,'ij,jk->ik').mat, ChoiUV.full)
    
    # Test the ``@`` operator
    assert np.allclose((ChoiU @ ChoiV).mat, ChoiUV.full)


# %%
