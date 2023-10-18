import collections

import numpy as np
import sympy

from qibo import gates
from qibo.config import raise_error
from qibo.states import QuantumState


def frequencies_to_binary(frequencies, nqubits):
    return collections.Counter(
        {"{:b}".format(k).zfill(nqubits): v for k, v in frequencies.items()}
    )


def apply_bitflips(result, p0, p1=None):
    gate = result.measurement_gate
    if p1 is None:
        probs = 2 * (gate._get_bitflip_tuple(gate.qubits, p0),)
    else:
        probs = (
            gate._get_bitflip_tuple(gate.qubits, p0),
            gate._get_bitflip_tuple(gate.qubits, p1),
        )
    noiseless_samples = result.samples()
    return result.backend.apply_bitflips(noiseless_samples, probs)


class MeasurementSymbol(sympy.Symbol):
    """``sympy.Symbol`` connected to measurement results.

    Used by :class:`qibo.gates.measurements.M` with ``collapse=True`` to allow
    controlling subsequent gates from the measurement results.
    """

    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = "m{}".format(cls._counter)
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, index, result):
        self.index = index
        self.result = result

    def __getstate__(self):
        return {"index": self.index, "result": self.result, "name": self.name}

    def __setstate__(self, data):
        self.index = data.get("index")
        self.result = data.get("result")
        self.name = data.get("name")

    def outcome(self):
        return self.result.samples(binary=True)[-1][self.index]

    def evaluate(self, expr):
        """Substitutes the symbol's value in the given expression.

        Args:
            expr (sympy.Expr): Sympy expression that involves the current
                measurement symbol.
        """
        return expr.subs(self, self.outcome())


class MeasurementResult:
    """Data structure for holding measurement outcomes.

    :class:`qibo.measurements.MeasurementResult` objects can be obtained
    when adding measurement gates to a circuit.

    Args:
        gate (:class:`qibo.gates.M`): Measurement gate associated with
            this result object.
        nshots (int): Number of measurement shots.
        backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend
            to use for calculations.
    """

    def __init__(self, gate, nshots=0, backend=None):
        self.measurement_gate = gate
        self.backend = backend
        self.nshots = nshots
        self.circuit = None

        self._samples = None
        self._frequencies = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None
        self._symbols = None

    def __repr__(self):
        qubits = self.measurement_gate.qubits
        nshots = self.nshots
        return f"MeasurementResult(qubits={qubits}, nshots={nshots})"

    def add_shot(self, probs):
        qubits = sorted(self.measurement_gate.target_qubits)
        shot = self.backend.sample_shots(probs, 1)
        bshot = self.backend.samples_to_binary(shot, len(qubits))
        if self._samples:
            self._samples.append(bshot[0])
        else:
            self._samples = [bshot[0]]
        self.nshots += 1
        return shot

    def has_samples(self):
        return self._samples is not None

    def register_samples(self, samples, backend=None):
        """Register samples array to the ``MeasurementResult`` object."""
        self._samples = samples
        self.nshots = len(samples)

    def register_frequencies(self, frequencies, backend=None):
        """Register frequencies to the ``MeasurementResult`` object."""
        self._frequencies = frequencies
        self.nshots = sum(frequencies.values())

    def reset(self):
        """Remove all registered samples and frequencies."""
        self._samples = None
        self._frequencies = None

    @property
    def symbols(self):
        """List of ``sympy.Symbols`` associated with the results of the measurement.

        These symbols are useful for conditioning parametrized gates on measurement outcomes.
        """
        if self._symbols is None:
            qubits = self.measurement_gate.target_qubits
            self._symbols = [MeasurementSymbol(i, self) for i in range(len(qubits))]

        return self._symbols

    def samples(self, binary=True, registers=False):
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
        """
        if self._samples is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError, "Cannot calculate samples if circuit is not provided."
                )
            # calculate samples for the whole circuit so that
            # individual register samples are registered here
            self.circuit.final_state.samples()
        if binary:
            return np.array(self._samples, dtype="int32")
        else:
            qubits = self.measurement_gate.target_qubits
            return self.backend.samples_to_decimal(self._samples, len(qubits))

    def frequencies(self, binary=True, registers=False):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
        """
        if self._frequencies is None:
            self._frequencies = self.backend.calculate_frequencies(
                self.samples(binary=False)
            )
        if binary:
            qubits = self.measurement_gate.target_qubits
            return frequencies_to_binary(self._frequencies, len(qubits))
        else:
            return self._frequencies

    def apply_bitflips(self, p0, p1=None):  # pragma: no cover
        return apply_bitflips(self, p0, p1)


class MeasurementOutcomes:
    """Object to store the outcomes of measurements after circuit execution."""

    def __init__(
        self, measurements, backend, probabilities=None, samples=None, nshots=1000
    ):
        if probabilities is None and samples is None:
            raise RuntimeError(
                "You have to provide either the `probabilities` or the `samples` to build a `MeasurementOutcomes` object."
            )
        if probabilities is not None and samples is not None:
            raise RuntimeError(
                "Both the `probabilities` and the `samples` were provided to build the `MeasurementOutcomes` object. Don't know which one to use."
            )

        self.backend = backend
        self.measurements = measurements
        self.nshots = nshots

        self._measurement_gate = None
        self._probs = probabilities
        self._samples = samples
        self._frequencies = None
        self._repeated_execution_frequencies = None

        for gate in self.measurements:
            gate.result.reset()

    def frequencies(self, binary=True, registers=False):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
            If `registers` is `True`
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If `registers` is `False`
                a single `Counter` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        if self._repeated_execution_frequencies is not None:
            return self._repeated_execution_frequencies

        qubits = self.measurement_gate.qubits
        if self._frequencies is None:
            if self.measurement_gate.has_bitflip_noise() and not self.has_samples():
                self._samples = self.samples()
            if not self.has_samples():
                # generate new frequencies
                self._frequencies = self.backend.sample_frequencies(
                    self._probs, self.nshots
                )
                # register frequencies to individual gate ``MeasurementResult``
                qubit_map = {q: i for i, q in enumerate(qubits)}
                reg_frequencies = {}
                binary_frequencies = frequencies_to_binary(
                    self._frequencies, len(qubits)
                )
                for gate in self.measurements:
                    rfreqs = collections.Counter()
                    for bitstring, freq in binary_frequencies.items():
                        idx = 0
                        rqubits = gate.target_qubits
                        for i, q in enumerate(rqubits):
                            if int(bitstring[qubit_map.get(q)]):
                                idx += 2 ** (len(rqubits) - i - 1)
                        rfreqs[idx] += freq
                    gate.result.register_frequencies(rfreqs, self.backend)
            else:
                self._frequencies = self.backend.calculate_frequencies(
                    self.samples(binary=False)
                )

        if registers:
            return {
                gate.register_name: gate.result.frequencies(binary)
                for gate in self.measurements
            }

        if binary:
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def has_samples(self):
        if self._samples is not None:
            assert self.measurements[0].result.has_samples()
            return True
        else:  # pragma: no cover
            return False

    def samples(self, binary=True, registers=False):
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
            If `registers` is `True`
                samples are returned in a `dict` where the keys are the register
                names and the values are the samples tensors for each register.
            If `registers` is `False`
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        qubits = self.measurement_gate.target_qubits
        if self._samples is None:
            if self.measurements[0].result.has_samples():
                self._samples = np.concatenate(
                    [gate.result.samples() for gate in self.measurements], axis=1
                )
            else:
                if self._frequencies is not None:
                    # generate samples that respect the existing frequencies
                    frequencies = self.frequencies(binary=False)
                    samples = np.concatenate(
                        [np.repeat(x, f) for x, f in frequencies.items()]
                    )
                    np.random.shuffle(samples)
                else:
                    # generate new samples
                    samples = self.backend.sample_shots(self._probs, self.nshots)
                samples = self.backend.samples_to_binary(samples, len(qubits))
                if self.measurement_gate.has_bitflip_noise():
                    p0, p1 = self.measurement_gate.bitflip_map
                    bitflip_probabilities = [
                        [p0.get(q) for q in qubits],
                        [p1.get(q) for q in qubits],
                    ]
                    samples = self.backend.apply_bitflips(
                        samples, bitflip_probabilities
                    )
                # register samples to individual gate ``MeasurementResult``
                qubit_map = {
                    q: i for i, q in enumerate(self.measurement_gate.target_qubits)
                }
                self._samples = np.array(samples, dtype="int32")
                for gate in self.measurements:
                    rqubits = tuple(qubit_map.get(q) for q in gate.target_qubits)
                    gate.result.register_samples(
                        self._samples[:, rqubits], self.backend
                    )

        if registers:
            return {
                gate.register_name: gate.result.samples(binary)
                for gate in self.measurements
            }

        if binary:
            return self._samples
        else:
            return self.backend.samples_to_decimal(self._samples, len(qubits))

    @property
    def measurement_gate(self):
        """Single measurement gate containing all measured qubits.

        Useful for sampling all measured qubits at once when simulating.
        """
        if self._measurement_gate is None:
            for gate in self.measurements:
                if self._measurement_gate is None:
                    self._measurement_gate = gates.M(
                        *gate.init_args, **gate.init_kwargs
                    )
                else:
                    self._measurement_gate.add(gate)

        return self._measurement_gate

    def apply_bitflips(self, p0, p1=None):
        return apply_bitflips(self, p0, p1)

    def expectation_from_samples(self, observable):
        """Computes the real expectation value of a diagonal observable from frequencies.

        Args:
            observable (Hamiltonian/SymbolicHamiltonian): diagonal observable in the
            computational basis.

        Returns:
            Real number corresponding to the expectation value.
        """
        freq = self.frequencies(binary=True)
        qubit_map = self.measurement_gate.qubits
        return observable.expectation_from_samples(freq, qubit_map)

    def dumps(self):
        args = {
            "measurements": [m.to_json for m in self.measurements],
            "backend": self.backend.name,
            "platform": self.backend.platform,
            "probabilities": self._probs,
            "samples": self._samples,
            "nshots": self.nshots,
        }
        return args

    def dump(self, filename):
        np.save(filename, self.dump())

    @classmethod
    def load(cls, filename):
        load = np.load(filename, allow_pickle=True).item()
        backend = backends.construct_backend(load.get("backend"), load.get("platform"))
        mesaurements = []
        for m in load.get("measurements"):
            args = json.loads(m)
            qubits = m.pop("_target_qubits")
            args["basis"] = getattr(gates, args["basis"])
            measurements.append(gates.M(*qubits, **args))
        return cls(
            measurements,
            backend,
            probabilities=load.get("probabilities"),
            samples=load.get("samples"),
            nshots=load.get("nshots"),
        )


class CircuitResult(QuantumState, MeasurementOutcomes):
    """Object to store both the outcomes of measurements and the final state after circuit execution."""

    def __init__(self, final_state, measurements, backend, nshots=1000):
        QuantumState.__init__(self, final_state, backend)
        qubits = [q for m in measurements for q in m.target_qubits]
        if len(qubits) == 0:
            raise ValueError(
                "Circuit does not contain measurements. Use a `QuantumState` instead."
            )
        probs = self.probabilities(qubits)
        MeasurementOutcomes.__init__(
            self, measurements, backend, probabilities=probs, nshots=nshots
        )
