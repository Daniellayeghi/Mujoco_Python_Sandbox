import matplotlib.pyplot as plt
import numpy as np
from recordtype import recordtype

GridParams = recordtype(
    'GridParams',
    'row_col def_cost min_cost cost_spread goal'
)


class GridWorld:
    def __init__(self, gp: GridParams):
        self.gp = gp
        self._grid = np.ones((gp.row_col[0], gp.row_col[1]))
        self._actions = {"up": [-1, 0], "down": [+1, 0], "left": [0, -1], "right": [0, +1]}

    def cost_grid(self):
        self._grid = self._grid * self.gp.def_cost
        for idx, cost in self.gp.cost_spread.items():
            self._grid[idx[0], idx[1]] = cost

        return self._grid

    def step(self, action: str, state: np.array):
        valid = lambda s: (0 <= s[0] < self.gp.row_col[0]) and \
                          (0 <= s[1] < self.gp.row_col[1])

        if np.linalg.norm(state-self.gp.goal) == 0.0:
            return state, 0

        cost = self._grid[state[0]][state[1]]
        act = self._actions[action]
        new_state = state + act
        if valid(new_state):
            state = new_state

        return state, cost


class ValueIteration:
    def __init__(self, world: GridWorld, actions: list, states: list, it_lim: int):
        self._world = world
        self._actions = actions
        self._states = states
        self._values = np.zeros_like(self._world.cost_grid())
        self._iteration_limit = it_lim

    def compute_values(self):
        for iteration in range(self._iteration_limit):
            for s1 in self._states[0]:
                for s2 in self._states[1]:
                    v = []
                    for a in self._actions:
                        state, cost = self._world.step(a, np.array([int(s1), int(s2)]))
                        n_v = cost + self._values[int(state[0]), int(state[1])]
                        v.append(n_v)

                    self._values[int(s1), int(s2)] = min(v)

        return self._values


def plot_value(values: np.array, min_max: list, title: str):
    plt.figure()
    plt.imshow(
        values,
        interpolation=None,
        extent=[min_max[0], min_max[1]+1, min_max[1]+1, min_max[0]]
    )
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    # Define world constant
    row, col = 10, 10
    total_size = row * col
    x = np.linspace(0, row - 1, row)
    y = np.linspace(0, col - 1, col)
    states = [x.tolist(), y.tolist()]
    actions = ["up", "down", "left", "right"]

    ex_per_sample = 1
    samples = int(row*col * 0.75)
    results = np.zeros((total_size, samples * ex_per_sample))
    perts = np.random.randint(1, 10, ex_per_sample)
    gp = GridParams(
            row_col=[row, col],
            def_cost=1,
            min_cost=0,
            cost_spread={(0, 0): 0},
            goal=np.array([0, 0])
        )

    gw = GridWorld(gp)

    for i in range(samples):
        # Sample IID goals
        min_bound, max_bound = 0, col - 1
        goal = np.random.randint(min_bound, max_bound, 2)
        for j in range(ex_per_sample):
            # Update gw
            gp.def_cost = 1
            gp.cost_spread = {(goal[0], goal[1]): 0}
            gp.goal = goal
            gw = GridWorld(gp)
            # Compute value map
            vi = ValueIteration(gw, actions, states, 40)
            values = vi.compute_values()
            results[:, i*ex_per_sample + j] = values.reshape(np.size(values))