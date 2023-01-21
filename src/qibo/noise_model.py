import numpy as np
from scipy.linalg import sqrtm

from qibo import gates, models
from qibo.quantum_info import hellinger_distance, hellinger_fidelity


def noisy_circuit(circuit, params):
    """Creates a noisy circuit from the circuit given as argument.

    The function applies a :class:`qibo.gates.ThermalRelaxationChannel` after each step of the circuit
    and, after each gate, a :class:`qibo.gates.DepolarizingChannel`, whose parameter depends on whether the
    gate applies on one or two qubits. In the end are applied  asymmetric bitflips on measurement gates.


    Args:
        circuit (qibo.models.Circuit): Circuit on which noise will be applied. Since in the end are
        applied bitflips, measurement gates are required.
        params (dictionary): object which contains the parameters of the channels organized as follow
        params = {"t1" : (t1, t2,..., tn),
          "t2" : (t1, t2,..., tn),
          "gate time" : (time1, time2),
          "excited population" : 0,
          "depolarizing error" : (lambda1, lambda2),
          "bitflips error" : ([p1, p2,..., pm], [p1, p2,..., pm]),
          "idle_qubits" : 1
         }
        Where n is the number of qubits, and m the number of measurement gates.
        The first four parameters are used by the thermal relaxation error. The first two  elements are the
        tuple containing the T_1 and T_2 parameters; the third one is a tuple which contain the gate times,
        for single and two qubit gates; then we have the excited population parameter.
        The fifth parameter is a tuple containing the depolaraziong errors for single and 2 qubit gate.
        The sisxth parameter is a tuple containg the two arrays for bitflips probability errors: the first one implements 0->1 errors, the other one 1->0.
        The last parameter is a boolean variable: if True the noise model takes into account idle qubits.

    Returns:
        The new noisy circuit (qibo.models.Circuit).


    """
    # parameters of the model
    t1 = params["t1"]
    t2 = params["t2"]
    time1 = params["gate_time"][0]
    time2 = params["gate_time"][1]
    excited_population = params["excited_population"]
    depolarizing_error_1 = params["depolarizing_error"][0]
    depolarizing_error_2 = params["depolarizing_error"][1]
    bitflips_01 = params["bitflips_error"][0]
    bitflips_10 = params["bitflips_error"][1]
    idle_qubits = params["idle_qubits"]

    # new circuit
    noisy_circ = models.Circuit(circuit.nqubits, density_matrix=True)

    # time steps of the circuit
    time_steps = max(circuit.queue.moment_index)

    # current_time keeps track of the time spent by the qubits
    # being manipulated by the gates of the circuit
    current_time = np.zeros(circuit.nqubits)

    # the idea behind ths loop is to build the old circuit adding the noise channels and
    # keeping track of the time qubits spend being manipulated by the gates, in order
    # to correct the thermal relaxation time of each qubit, even if they are idle.
    for t in range(time_steps):
        # for each time step, I look for each qubit what gate are applied
        for qubit in range(circuit.nqubits):
            # if there's no gate, move on!
            if circuit.queue.moments[t][qubit] == None:
                pass
            # measurement gates
            elif isinstance(circuit.queue.moments[t][qubit], gates.measurements.M):
                for key in list(circuit.measurement_tuples):
                    # if there is a 2-qubits measurement gate we must check that both qubit intercated
                    # with the environment for the same amount of time. If not, before applying
                    # the 2-qubits gate we apply the therm-rel channel for the time difference
                    if len(circuit.measurement_tuples[key]) > 1:
                        q1 = circuit.measurement_tuples[key][0]
                        q2 = circuit.measurement_tuples[key][1]
                        if current_time[q1] != current_time[q2] and idle_qubits == True:
                            q_min = q1
                            q_max = q2
                            if current_time[q1] > current_time[q2]:
                                q_min = q2
                                q_max = q1
                            time_difference = current_time[q_max] - current_time[q_min]
                            # this is the thermal relaxation channel which model the intercation
                            # of the idle qubit with the environment
                            noisy_circ.add(
                                gates.ThermalRelaxationChannel(
                                    q_min,
                                    t1[q_min],
                                    t2[q_min],
                                    time_difference,
                                    excited_population,
                                )
                            )
                            # update the qubit time
                            current_time[q_min] += time_difference
                q = circuit.queue.moments[t][qubit].qubits
                # adding measurements gates
                if len(circuit.queue.moments[t][qubit].qubits) == 1:
                    q = q[0]
                    noisy_circ.add(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
                else:
                    p0q = []
                    p1q = []
                    for j in q:
                        p0q.append(bitflips_01[j])
                        p1q.append(bitflips_10[j])
                    noisy_circ.add(gates.M(*q, p0=p0q, p1=p1q))
                    circuit.queue.moments[t][
                        max(circuit.queue.moments[t][qubit].qubits)
                    ] = None
            # if there is a 1-qubit gate I add the old gate, the dep and therm-rel channels
            elif len(circuit.queue.moments[t][qubit].qubits) == 1:
                noisy_circ.add(circuit.queue.moments[t][qubit])
                noisy_circ.add(
                    gates.DepolarizingChannel(
                        circuit.queue.moments[t][qubit].qubits, depolarizing_error_1
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        qubit,
                        t1[qubit],
                        t2[qubit],
                        time1,
                        excited_population,
                    )
                )
                # I update the qubit time
                current_time[qubit] += time1
            # if there is a 2-qubits gate we must check that both qubit intercated
            # with the environment for the same amount of time. If not, before applying
            # the 2-qubits gate we apply the therm-rel channel for the time difference
            else:
                q1 = circuit.queue.moments[t][qubit].qubits[0]
                q2 = circuit.queue.moments[t][qubit].qubits[1]
                if current_time[q1] != current_time[q2] and idle_qubits == True:
                    q_min = q1
                    q_max = q2
                    if current_time[q1] > current_time[q2]:
                        q_min = q2
                        q_max = q1
                    time_difference = current_time[q_max] - current_time[q_min]
                    # this is the thermal relaxation channel which model the intercation
                    # of the idle qubit with the environment
                    noisy_circ.add(
                        gates.ThermalRelaxationChannel(
                            q_min,
                            t1[q_min],
                            t2[q_min],
                            time_difference,
                            excited_population,
                        )
                    )
                    # I update the qubit time
                    current_time[q_min] += time_difference
                # I add the 2-qubit gate, dep and therm-rel channels
                noisy_circ.add(circuit.queue.moments[t][qubit])
                noisy_circ.add(
                    gates.DepolarizingChannel(
                        tuple(set(circuit.queue.moments[t][qubit].qubits)),
                        depolarizing_error_2,
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        q1, t1[q1], t2[q1], time2, excited_population
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        q2, t1[q2], t2[q2], time2, excited_population
                    )
                )
                # I update the qubit time
                current_time[circuit.queue.moments[t][qubit].qubits[0]] += time2
                current_time[circuit.queue.moments[t][qubit].qubits[1]] += time2
                circuit.queue.moments[t][
                    max(circuit.queue.moments[t][qubit].qubits)
                ] = None

    measurements = []
    for m in circuit.measurements:
        q = m.qubits
        if len(q) == 1:
            q = q[0]
            measurements.append(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
        else:
            p0q = []
            p1q = []
            for j in q:
                p0q.append(bitflips_01[j])
                p1q.append(bitflips_10[j])
            measurements.append(gates.M(*q, p0=p0q, p1=p1q))
    noisy_circ.measurements = measurements

    return noisy_circ


def freq_to_prob(freq):
    norm = sum(freq.values())
    nqubits = len(list(freq.keys())[0])
    prob = np.zeros(2**nqubits)
    for k in range(2**nqubits):
        index = "{:b}".format(k).zfill(nqubits)
        prob[k] = freq[index] / norm
    return prob


def hellinger_shot_error(p, q, nshots):
    prob_p = np.sqrt((p - p**2) / nshots)
    prob_q = np.sqrt((q - q**2) / nshots)
    hellinger_dist = hellinger_distance(p, q)
    hellinger_dist_e = np.sum(
        (abs(1 - np.sqrt(q / p)) * prob_p + abs(1 - np.sqrt(p / q)) * prob_q)
        / (4 * hellinger_dist)
    )
    hellinger_fid_e = 4 * hellinger_dist * (1 - hellinger_dist**2) * hellinger_dist_e
    return hellinger_fid_e


def loss(parameters, grad, args):
    circuit = args[0]
    nshots = args[1]
    target_prob = args[2]
    idle_qubits = args[3]
    backend = args[4]
    error = args[5]
    qubits = circuit.nqubits
    parameters = np.array(parameters)

    if any(2 * parameters[0:qubits] - parameters[qubits : 2 * qubits] < 0):
        return -np.inf

    params = {
        "t1": tuple(parameters[0:qubits]),
        "t2": tuple(parameters[qubits : 2 * qubits]),
        "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
        "excited_population": 0,
        "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
        "bitflips_error": (
            parameters[2 * qubits + 4 : 3 * qubits + 4],
            parameters[3 * qubits + 4 : 4 * qubits + 4],
        ),
        "idle_qubits": idle_qubits,
    }

    noisy_circ = noisy_circuit(circuit, params)
    freq = backend.execute_circuit(circuit=noisy_circ, nshots=nshots).frequencies()
    prob = freq_to_prob(freq)

    hellinger_fid = hellinger_fidelity(target_prob, prob)

    if error == True:
        return [hellinger_fid, hellinger_shot_error(target_prob, prob, nshots)]
    else:
        return hellinger_fid


class CompositeNoiseModel:
    def __init__(self, params):
        self.noisy_circuit = {}
        self.params = params
        self.hellinger = {}
        self.hellinger0 = {}

    def apply(self, circuit):
        self.noisy_circuit = noisy_circuit(circuit, self.params)

    def fit(
        self,
        target_result,
        bounds=True,
        f_min_rtol=None,
        backend=None,
    ):
        from functools import partial

        import nlopt

        if backend == None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        circuit = target_result.circuit
        nshots = target_result.nshots
        target_prob = freq_to_prob(target_result.frequencies())

        idle_qubits = self.params["idle_qubits"]
        qubits = target_result.nqubits

        if bounds == True:
            qubits = target_result.nqubits
            lb = np.zeros(4 * qubits + 4)
            ub = [10000] * (2 * qubits + 2) + [4 / 3, 16 / 15] + [1] * 2 * qubits
        else:
            lb = bounds[0]
            ub = bounds[1]

        shot_error = True
        args = (circuit, nshots, target_prob, idle_qubits, backend, shot_error)
        result = -np.inf
        while result == -np.inf:
            initial_params = np.random.uniform(lb, ub)
            result = loss(initial_params, 0, args)

        if f_min_rtol == None:
            f_min_rtol = result[1]

        args = list(args)
        args[5] = False
        args = tuple(args)

        self.hellinger0 = {"fidelity": abs(result[0]), "shot_error": result[1]}

        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, len(initial_params))
        f = partial(loss, args=args)
        opt.set_max_objective(f)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_stopval(1 - f_min_rtol)
        xopt = opt.optimize(list(initial_params))
        maxf = opt.last_optimum_value()
        result = opt.last_optimize_result()

        parameters = xopt
        params = {
            "t1": tuple(parameters[0:qubits]),
            "t2": tuple(parameters[qubits : 2 * qubits]),
            "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
            "excited_population": 0,
            "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
            "bitflips_error": (
                parameters[2 * qubits + 4 : 3 * qubits + 4],
                parameters[3 * qubits + 4 : 4 * qubits + 4],
            ),
            "idle_qubits": idle_qubits,
        }
        self.hellinger = maxf
        self.params = params
        self.extra = {
            "message": result,
        }
