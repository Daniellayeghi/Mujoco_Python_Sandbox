import time
import mujoco
import concurrent.futures
import numpy as np
from mujoco import rollout


class ParallelRollouts:
    def __init__(self, model_path: str, nworkers):
        self._nworkers = nworkers
        self._m = mujoco.MjModel.from_xml_path(model_path)
        self._chunks = None

    def _rollout(self, init_states, ctrl, state, data):
        nstate, nstep, _ = state.shape
        rollout.rollout(
            self._m, data, skip_checks=True, nstate=nstate,
            nstep=nstep, initial_state=init_states, ctrl=ctrl, state=state
        )

    def _build_containers(self, init_states, state, ctrl):
        n = init_states.shape[0] // self._nworkers  # integer division
        self._chunks = [] # a list of tuples, one per worker
        for i in range(self._nworkers - 1):
            self._chunks.append([
                    init_states[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n],
                    state[i * n:(i + 1) * n], mujoco.MjData(self._m)
                ])

        # last chunk, absorbing the remainder:
        self._chunks.append([
                init_states[(self._nworkers - 1) * n:], ctrl[(self._nworkers - 1) * n:],
                state[(self._nworkers - 1) * n:], mujoco.MjData(self._m)
            ])

    def _fill_containers(self, init_states, state, ctrl):
        n = init_states.shape[0] // self._nworkers  # integer division
        for i in range(self._nworkers - 1):
            self._chunks[i][0], self._chunks[i][1], self._chunks[i][2] =\
                init_states[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n], state[i * n:(i + 1) * n]
        # last chunk, absorbing the remainder:
        self._chunks[-1][0], self._chunks[-1][1], self._chunks[-1][2] =\
            init_states[(self._nworkers - 1) * n:], ctrl[(self._nworkers - 1) * n:], state[(self._nworkers - 1) * n:]

    def __call__(self, init_states, state, ctrl, sensor_data, use_cache=False):
        prepare_containers = self._fill_containers if use_cache else self._build_containers
        prepare_containers(init_states, state, ctrl)
        assert(self._chunks is not None)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._nworkers, initializer=None) as executor:
            futures = []
            for chunk in self._chunks:
                futures.append(executor.submit(self._rollout, *chunk))
            for future in concurrent.futures.as_completed(futures):
                future.result()

        return state


if __name__ == "__main__":
    model_path = "../../../OptimisationBasedControl/models/finger.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    num_workers, nstate, nstep = 10, 100, 200
    initial_state = np.random.randn(nstate, model.nq + model.nv + model.na)
    state = np.zeros((nstate, nstep, model.nq + model.nv + model.na))
    sensordata = np.zeros((nstate, nstep, model.nsensordata))
    ctrl = np.random.randn(nstate, nstep, model.nu)

    data = mujoco.MjData(model)
    par_roll = ParallelRollouts(model_path, num_workers)
    res = par_roll(initial_state, state, ctrl, sensordata, use_cache=False)
    res_cached = par_roll(initial_state, state, ctrl, sensordata, use_cache=True)
    print(res_cached)
