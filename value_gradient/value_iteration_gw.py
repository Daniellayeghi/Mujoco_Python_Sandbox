import numpy as np
from collections import namedtuple

GridParams = namedtuple(
    'GridParams',
    'row_col def_cost min_cost cost_spread goal'
)


class GridWorld:
    def __init__(self, gp: GridParams):
        self._gp = gp
        self._grid = np.ones((gp.row_col[0], gp.row_col[1]))
        self._actions = {"up": [-1, 0], "down": [+1, 0], "left": [0, -1], "right": [0, +1]}

    def cost_grid(self):
        self._grid = self._grid * self._gp.def_cost
        for idx, cost in self._gp.cost_spread.items():
            self._grid[idx[0], idx[1]] = cost

        return self._grid

    def step(self, action: str, state: np.array):
        valid = lambda s: (0 <= s[0] < self._gp.row_col[0]) and \
                          (0 <= s[1] < self._gp.row_col[1])

        if np.linalg.norm(state-self._gp.goal) == 0.0:
            return state, 0

        cost = self._grid[state[0]][state[1]]
        act = self._actions[action]
        new_state = state + act
        if valid(new_state):
            state = new_state

        return state, cost


class ValueIteration:
    def __init__(self, world: GridWorld, actions: list, states: list):
        self._world = world
        self._actions = actions
        self._states = states
        self._values = np.zeros_like(self._world.cost_grid())

    def compute_values(self):
        for iteration in range(10):
            for s1 in self._states[0]:
                for s2 in self._states[1]:
                    v = []
                    for a in self._actions:
                        state, cost = self._world.step(a, np.array([s1, s2]))
                        n_v = cost + self._values[state[0], state[1]]
                        v.append(n_v)

                    self._values[s1, s2] += min(v)

        return self._values


if __name__ == "__main__":
    gp = GridParams(
        row_col=[5, 5],
        def_cost=1,
        min_cost=0,
        cost_spread={(0, 0): 0},
        goal=np.array([0, 0])
    )

    gw = GridWorld(gp)
    actions = ["up", "down", "left", "right"]
    states = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    vi = ValueIteration(gw, actions, states)
    values = vi.compute_values()
    print(f"values : {values}")
