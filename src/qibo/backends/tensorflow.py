import os
from qibo.backends.abstract import AbstractBackend, AbstractCustomOperators
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error, log, TF_LOG_LEVEL, TF_MIN_VERSION


class Optimization:

    def __init__(self):
        import tensorflow as tf  # pylint: disable=E0401
        self.Variable = tf.Variable
        self.GradientTape = tf.GradientTape
        self.optimizers = tf.optimizers


class TensorflowBackend(NumpyBackend):

    description = "Uses `tf.einsum` to apply gates to states via matrix " \
                  "multiplication."

    def __init__(self):
        super().__init__()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)
        import tensorflow as tf  # pylint: disable=E0401
        if tf.__version__ < TF_MIN_VERSION:  # pragma: no cover
            raise_error(RuntimeError, "TensorFlow version not supported, "
                                      f"minimum is {TF_MIN_VERSION}.")
        self.backend = tf
        self.name = "tensorflow"

        self.cpu_devices = tf.config.list_logical_devices("CPU")
        self.gpu_devices = tf.config.list_logical_devices("GPU")
        if self.gpu_devices: # pragma: no cover
            # CI does not use GPUs
            self.default_device = self.gpu_devices[0].name
        elif self.cpu_devices:
            self.default_device = self.cpu_devices[0].name
        else: # pragma: no cover
            # case not tested by GitHub workflows because it requires no device
            raise_error(RuntimeError, "Unable to find Tensorflow devices.")

        self.tensor_types = (self.np.ndarray, tf.Tensor, tf.Variable)
        self.native_types = (tf.Tensor, tf.Variable)
        self.Tensor = tf.Tensor
        self.random = tf.random
        self.newaxis = tf.newaxis
        from tensorflow.python.framework import errors_impl  # pylint: disable=E0611,E0401
        self.oom_error = errors_impl.ResourceExhaustedError
        self.optimization = Optimization()

        # seed to use in the measurement frequency custom op
        self._seed = None
        # seed can be modified using ``K.set_seed``

        self.supports_gradients = True

    def set_device(self, name):
        AbstractBackend.set_device(self, name)

    def set_threads(self, nthreads):
        log.warning("`set_threads` is not supported by the tensorflow "
                    "backend. Please use tensorflow's thread setters: "
                    "`tf.config.threading.set_inter_op_parallelism_threads` "
                    "or `tf.config.threading.set_intra_op_parallelism_threads` "
                    "to switch the number of threads.")
        AbstractBackend.set_threads(self, nthreads)

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        return x.numpy()

    def to_complex(self, re, img):
        return self.backend.complex(re, img)

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.np.ndarray):
            dtypestr = dtype.__repr__().split(".")[1]
            x = x.astype(getattr(self.np, dtypestr))
        return self.backend.cast(x, dtype=dtype)

    def diag(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.cast(self.backend.linalg.diag(x), dtype=dtype)

    def concatenate(self, x, axis=None):
        return self.backend.concat(x, axis=axis)

    def copy(self, x):
        return x + self.backend.zeros_like(x)

    def range(self, start, finish, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.range(start, finish, step, dtype=dtype)

    def real(self, x):
        return self.backend.math.real(x)

    def imag(self, x):
        return self.backend.math.imag(x)

    def conj(self, x):
        return self.backend.math.conj(x)

    def mod(self, x, y):
        return self.backend.math.mod(x, y)

    def right_shift(self, x, y):
        return self.backend.bitwise.right_shift(x, y)

    def pow(self, base, exponent):
        return self.backend.math.pow(base, exponent)

    def square(self, x):
        return self.backend.square(x)

    def sqrt(self, x):
        return self.backend.math.sqrt(x)

    def log(self, x):
        return self.backend.math.log(x)

    def trace(self, x):
        return self.backend.linalg.trace(x)

    def expm(self, x):
        return self.backend.linalg.expm(x)

    def sum(self, x, axis=None):
        return self.backend.reduce_sum(x, axis=axis)

    def outer(self, x, y):
        return self.tensordot(x, y, axes=0)

    def kron(self, x, y):
        dim = int(x.shape[0]) * int(y.shape[0])
        z = self.transpose(self.outer(x, y), axes=[0, 2, 1, 3])
        return self.reshape(z, (dim, dim))

    def inv(self, x):
        raise_error(NotImplementedError)

    def unique(self, x, return_counts=False):
        if return_counts:
            res, _, counts = self.backend.unique_with_counts(
                x, out_idx=self.dtypes('DTYPEINT'))
            return res, counts
        res, _  = self.backend.unique(x, out_idx=self.dtypes('DTYPEINT'))
        return res

    def gather(self, x, indices=None, condition=None, axis=0):
        if indices is not None:
            return self.backend.gather(x, indices, axis=axis)

        if condition is None:
            raise_error(ValueError, "Gather call is missing indices and "
                                    "condition.")
        indices = self.backend.where(condition)
        return self.backend.gather(x, indices, axis=axis)[:, 0]

    def gather_nd(self, x, indices):
        return self.backend.gather_nd(x, indices)

    def initial_state(self, nqubits, is_matrix=False): # pragma: no cover
        dim = 1 + is_matrix
        shape = dim * (2 ** nqubits,)
        idx = self.backend.constant([dim * [0]], dtype=self.dtypes('DTYPEINT'))
        state = self.backend.zeros(shape, dtype=self.dtypes('DTYPECPX'))
        update = self.backend.constant([1], dtype=self.dtypes('DTYPECPX'))
        state = self.backend.tensor_scatter_nd_update(state, idx, update)
        return state

    def random_uniform(self, shape, dtype='DTYPE'):
        return self.backend.random.uniform(shape, dtype=self.dtypes(dtype))

    def sample_shots(self, probs, nshots):
        from qibo.config import SHOT_BATCH_SIZE
        logits = self.log(probs)[self.newaxis]
        samples = [self.random.categorical(
            logits, SHOT_BATCH_SIZE, dtype=self.dtypes('DTYPEINT'))[0]
            for _ in range(nshots // SHOT_BATCH_SIZE)]
        samples.append(self.random.categorical(
                logits, nshots % SHOT_BATCH_SIZE,
                dtype=self.dtypes('DTYPEINT'))[0])
        return self.concatenate(samples, axis=0)

    def sample_frequencies(self, probs, nshots):
        logits = self.log(probs)[self.newaxis]
        samples = self.random.categorical(logits, nshots, dtype=self.dtypes('DTYPEINT'))[0]
        res, counts = self.unique(samples, return_counts=True)
        frequencies = self.zeros(int(probs.shape[0]), dtype=self.dtypes('DTYPEINT'))
        frequencies = self.backend.tensor_scatter_nd_add(frequencies, res[:, self.newaxis], counts)
        return frequencies

    def compile(self, func):
        return self.backend.function(func)

    def device(self, device_name):
        return self.backend.device(device_name)

    def on_cpu(self):
        return self.device(self.cpu_devices[0])

    def cpu_tensor(self, x, dtype=None):
        if dtype is None:
            dtype = x.dtype
        return self.backend.Variable(x, dtype=dtype)

    def cpu_cast(self, x, dtype='DTYPECPX'):
        dtype = self._dtypes.get(dtype)
        with self.on_cpu():
            return self.cast(x, dtype=dtype)

    def cpu_assign(self, state, i, piece):
        state.pieces[i].assign(piece)

    def executing_eagerly(self):
        return self.backend.executing_eagerly()

    def set_seed(self, seed):
        self._seed = seed
        self.backend.random.set_seed(seed)


class TensorflowCustomBackend(TensorflowBackend, AbstractCustomOperators):
    # TODO: Move this to qibotf repo

    description = "Uses precompiled primitives to apply gates to states. " \
                  "This is the fastest simulation engine."

    def __init__(self):
        TensorflowBackend.__init__(self)
        AbstractCustomOperators.__init__(self)
        from qibotf import custom_operators as op  # pylint: disable=E0401
        self.name = "qibotf"
        self.is_custom = True
        self.op = op
        if "OMP_NUM_THREADS" in os.environ: # pragma: no cover
            self.set_threads(int(os.environ.get("OMP_NUM_THREADS")))

        # enable multi-GPU if no macos
        import sys
        if sys.platform != "darwin":
            self.supports_multigpu = True

        # no gradient support for custom operators
        self.supports_gradients = False

    def set_threads(self, nthreads):
        AbstractBackend.set_threads(self, nthreads)

    def initial_state(self, nqubits, is_matrix=False):
        return self.op.initial_state(nqubits, self.dtypes('DTYPECPX'),
                                    is_matrix=is_matrix,
                                    omp_num_threads=self.nthreads)

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probs, nshots)
        # Generate random seed using tf
        dtype = self.dtypes('DTYPEINT')
        seed = self.backend.random.uniform(
            shape=tuple(), maxval=int(1e8), dtype=dtype)
        nqubits = int(self.np.log2(tuple(probs.shape)[0]))
        shape = self.cast(2 ** nqubits, dtype='DTYPEINT')
        frequencies = self.zeros(shape, dtype='DTYPEINT')
        frequencies = self.op.measure_frequencies(
            frequencies, probs, nshots, nqubits, seed, self.nthreads)
        return frequencies

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError)

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError)

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        qubits = [gate.nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(gate.nqubits - q - 1 for q in gate.target_qubits)
        cache.qubits_tensor = self.cast(sorted(qubits), "int32")
        if gate.density_matrix:
            cache.target_qubits_dm = [q + gate.nqubits for q in gate.target_qubits]
        return cache

    def _state_vector_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.cache.qubits_tensor, gate.nqubits, gate.target_qubits)

    def state_vector_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor, # pylint: disable=E1121
                       gate.nqubits, gate.target_qubits)

    def _density_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        state = gate_op(state, gate.cache.qubits_tensor + gate.nqubits, 2 * gate.nqubits, gate.target_qubits)
        state = gate_op(state, gate.cache.qubits_tensor, 2 * gate.nqubits, gate.cache.target_qubits_dm)
        return state

    def density_matrix_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        state = gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor + gate.nqubits, # pylint: disable=E1121
                        2 * gate.nqubits, gate.target_qubits)
        adjmatrix = self.conj(gate.custom_op_matrix)
        state = gate_op(state, adjmatrix, gate.cache.qubits_tensor,
                        2 * gate.nqubits, gate.cache.target_qubits_dm)
        return state

    def _density_matrix_half_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.cache.qubits_tensor + gate.nqubits, 2 * gate.nqubits, gate.target_qubits)

    def density_matrix_half_matrix_call(self, gate, state):
        gate_op = self.get_gate_op(gate)
        return gate_op(state, gate.custom_op_matrix, gate.cache.qubits_tensor + gate.nqubits, # pylint: disable=E1121
                       2 * gate.nqubits, gate.target_qubits)

    def _result_tensor(self, result):
        n = len(result)
        result = sum(2 ** (n - i - 1) * r for i, r in enumerate(result))
        return self.cast(result, dtype="DTYPEINT")

    def state_vector_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        return self.op.collapse_state(state, gate.cache.qubits_tensor, result,
                                      gate.nqubits, True, self.nthreads)

    def density_matrix_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        qubits = gate.cache.qubits_tensor
        state = self.op.collapse_state(state, qubits + gate.nqubits, result,
                                       2 * gate.nqubits, False, self.nthreads)
        state = self.op.collapse_state(state, qubits, result, 2 * gate.nqubits,
                                       False, self.nthreads)
        return state / self.trace(state)

    def compile(self, func):
        return func

    def transpose_state(self, pieces, state, nqubits, order):
        return self.op.transpose_state(pieces, state, nqubits, order, self.nthreads)

    def apply_gate(self, state, gate, qubits, nqubits, targets):
        return self.op.apply_gate(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_x(self, state, qubits, nqubits, targets):
        return self.op.apply_x(state, qubits, nqubits, *targets, self.nthreads)

    def apply_y(self, state, qubits, nqubits, targets):
        return self.op.apply_y(state, qubits, nqubits, *targets, self.nthreads)

    def apply_z(self, state, qubits, nqubits, targets):
        return self.op.apply_z(state, qubits, nqubits, *targets, self.nthreads)

    def apply_z_pow(self, state, gate, qubits, nqubits, targets):
        return self.op.apply_z_pow(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_two_qubit_gate(self, state, gate, qubits, nqubits, targets):
        return self.op.apply_two_qubit_gate(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_swap(self, state, qubits, nqubits, targets):
        return self.op.apply_swap(state, qubits, nqubits, *targets, self.nthreads)

    def apply_fsim(self, state, gate, qubits, nqubits, targets):
        return self.op.apply_fsim(state, gate, qubits, nqubits, *targets, self.nthreads)

    def apply_multiqubit_gate(self, state, gate, qubits, nqubits, targets):
        n = len(targets)
        raise_error(NotImplementedError,
                    "qibotf supports up to two-qubit gates but {} "
                    "targets were given. Please switch to another "
                    "backend to execute this operation.".format(n))

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        return self.op.collapse_state(state, qubits, result, nqubits, normalize, self.nthreads)

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        with self.on_cpu():
            return self.op.swap_pieces(piece0, piece1, new_global, nlocal, self.nthreads)
