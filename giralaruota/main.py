import logging
import numpy as np
import matplotlib.pyplot as plt


class GameResults():

    def __init__(self, n_groups):
        self.n_groups = n_groups
        self.population = np.empty((0, n_groups), int)
        self.hired = np.empty((0, n_groups), int)
        self.retired = np.empty((0, n_groups), int)
        self.score = np.empty((0, n_groups), float)
        self.expected_positions = np.empty((0, n_groups), float)

    def append(self, population, hired, retired,  score, expected_position):
        self.population = np.append(
            self.population, np.array([population]), axis=0)
        self.hired = np.append(self.hired, np.array([hired]), axis=0)
        self.retired = np.append(self.retired, np.array([retired]), axis=0)
        self.score = np.append(self.score, np.array([score]), axis=0)
        self.expected_positions = np.append(
            self.expected_positions, np.array([expected_position]), axis=0)


class GameMaster():
    ALGO_PROPORTIONAL = 'algo_proportional'
    ALGO_FLAT = 'algo_flat'
    ALGO_BALANCED = 'algo_balanced'

    def __init__(self, initial_population, retirement_rate, expected_hiring_per_year, hiring_algo):
        self._pop = np.array(initial_population)
        self._n_groups = len(initial_population)
        self.retirement_rate = retirement_rate
        self.expected_hiring_per_year = expected_hiring_per_year
        self._setup_basic_logging()
        self.hiring_algo = hiring_algo
        self._score = np.zeros(self._n_groups)
        self._update_group_hiring_per_year()
        self._reset_score()
        self.results = GameResults(self._n_groups)

    def _setup_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        logging.basicConfig(level=logging.DEBUG)
        self._logger = logging.getLogger('GameMaster')

    def _update_group_hiring_per_year(self):
        if self.hiring_algo == self.ALGO_PROPORTIONAL:
            self._ave_pos_year = self._group_hiring_per_year_proportional()
        elif self.hiring_algo == self.ALGO_FLAT:
            self._ave_pos_year = self._group_hiring_per_year_flat()
        elif self.hiring_algo == self.ALGO_BALANCED:
            self._ave_pos_year = self._group_hiring_per_year_balanced()
        else:
            raise ValueError('Unexpected algo %s' % self.hiring_algo)
        self._logger.debug("group hiring per year updated: %s" %
                           self._ave_pos_year)

    def _group_hiring_per_year_proportional(self):
        return self._pop / self.total_population * \
            self.expected_hiring_per_year

    def _group_hiring_per_year_flat(self):
        return np.ones(self._n_groups) / self._n_groups * \
            self.expected_hiring_per_year

    def _group_hiring_per_year_balanced(self):
        coeff = 0.7
        prop = self._group_hiring_per_year_proportional()
        flat = self._group_hiring_per_year_flat()
        return coeff*prop + (1-coeff)*flat

    def _reset_score(self):
        self._score += self._ave_pos_year
        self._logger.debug("score updated: %s" % self._score)

    @property
    def total_population(self):
        return self._pop.sum()

    def _compute_retired(self):
        return np.random.poisson(lam=self.retirement_rate*self._pop)

    def _who_is_next(self):
        nn = np.argmax(self._score)
        self._logger.debug("next is %d" % nn)
        return nn

    def _compute_hired(self):
        newpos = np.random.poisson(self.expected_hiring_per_year)
        self._logger.debug("hiring %d new positions" % newpos)
        hired = np.zeros(self._n_groups, dtype=int)
        for p in range(newpos):
            whosnext = self._who_is_next()
            hired[whosnext] += 1
            self._score[whosnext] -= 1
        return hired

    def _update_population(self, retired, hired):
        self._pop -= retired
        self._pop += hired
        self._pop = np.maximum(np.zeros(self._n_groups), self._pop)

    def run(self, years):
        for y in range(years):
            self._logger.info("step %d" % (y))
            self._logger.info("population %s" % (self._pop))
            self._logger.info("score %s" % (self._score))
            retired = self._compute_retired()
            self._logger.info("retired %s" % (retired))
            hired = self._compute_hired()
            self._logger.info("hired %s" % (hired))
            self.results.append(self._pop, hired, retired,
                                self._score, self._ave_pos_year)
            self._update_population(retired, hired)
            self._update_group_hiring_per_year()
            self._reset_score()


class Montecarlo():

    def __init__(self, how_many_runs, how_many_years, *args, **kwargs):
        self._n_runs = how_many_runs
        self._n_years = how_many_years
        self._pop_stat = None

        for i in range(self._n_runs):
            self.cycle(*args, **kwargs)

    def _update_results(self, results):
        if self._pop_stat is None:
            self._n_groups = results.n_groups
            self._pop_stat = np.empty(
                (0, self._n_years, self._n_groups), int)
        self._pop_stat = np.append(
            self._pop_stat, np.array([results.population]), axis=0)

    def cycle(self, *args, **kwargs):
        gm = GameMaster(*args, **kwargs)
        gm.run(self._n_years)
        self._update_results(gm.results)

    def plot(self):
        for g in range(self._n_groups):
            t = np.arange(self._n_years)
            val = self._pop_stat[:, :, g].mean(axis=0)
            err = self._pop_stat[:, :, g].std(axis=0)
            plt.plot(t, val, label='group %d' % (g+1))
            plt.fill_between(t, val-err, val+err,
                             alpha=0.5)
        val = self._pop_stat.sum(axis=2).mean(axis=0)
        err = self._pop_stat.sum(axis=2).std(axis=0)
        plt.plot(t, val, label='total')
        plt.fill_between(t, val-err, val+err,
                         alpha=0.5)

        plt.legend()
        plt.ylabel('Positions')
        plt.grid(True)

def main():
    st = Montecarlo(100, 20, [14, 20, 8, 8, 40], 1/30,
                    3, GameMaster.ALGO_PROPORTIONAL)
    plt.figure()
    st.plot()
    return st
