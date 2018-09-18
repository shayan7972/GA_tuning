import gym
import pandas as pd
import subprocess
import numpy as np
from gym import Env, spaces

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

CROSSOVER_RATE = [0, 0.2, 0.5, 0.8, 1]
POPULATION_SIZE = [4, 10, 50, 100, 200]
nrow, ncol = [5,5]


class GATuning(Env):
    goal_coverage = 1
    coverage_dict={} #for positions as key and its corresponding coverage as value

    def __init__(self):



        nA = 4
        nS = nrow * ncol
        self.action_space = spaces.Discrete(nA)
        # self.observation_space = spaces.Dict({"position": spaces.Discrete(nS), "coverage": spaces.Box(low=0,high=1,shape=(1,))})
        self.observation_space = spaces.Discrete(nS)
        self.reset()

        # isd = array.astype('float64').ravel()
        # isd /= isd.sum()
        #
        # P = {s: {a: [] for a in range(nA)} for s in range(nS)}







        # for row in range(nrow):
        #     for col in range(ncol):
        #         s = to_s(row, col)
        #         crossover = CROSSOVER_RATE[row]
        #         population = POPULATION_SIZE[col]
        #         for a in range(4):
        #             list = P[s][a]
        #             bc = b_coverage(crossover, population)
        #             if bc == 1:
        #                 list.append((1.0, s, 0, True))
        #             else:
        #                     newrow, newcol = inc(row, col, a)
        #                     newstate = to_s(newrow, newcol)
        #                     newcrossover = CROSSOVER_RATE[newrow]
        #                     newpop = POPULATION_SIZE[newcol]
        #                     newbc = b_coverage(newcrossover, newpop)
        #                     done = newbc == 1.0
        #                     reward = float(newbc == 1.0)
        #                     list.append((1.0, newstate, reward, done))

    def reset(self):
        position = 12
        self.state = position
        coverage = self.b_coverage(position)
        self.coverage_dict[position] = coverage
        return self.state


    def to_position(self, row, col):
        return row * ncol + col

    def to_coordinate(self, position):
        row = position // ncol
        col = position % ncol
        return row, col

    def inc(self, row, col, a):
        if a == 0:  # left
            col = max(col - 1, 0)
        elif a == 1:  # down
            row = min(row + 1, nrow - 1)
        elif a == 2:  # right
            col = min(col + 1, ncol - 1)
        elif a == 3:  # up
            row = max(row - 1, 0)
        return row, col

    def b_coverage(self, state):
        row, col = self.to_coordinate(state)
        crossover = CROSSOVER_RATE[row]
        population = POPULATION_SIZE[col]

        process = subprocess.Popen(
            ['java', '-jar', 'C:/Users/Shayan Z/Downloads/Tutorial_Stack/Tutorial_Stack/evosuite-1.0.6.jar',
             '-class', 'tutorial.Stack', '-projectCP', 'target/classes',
              '-Dcrossover_rate={}'.format(crossover), '-Dpopulation={}'.format(population),
             '-Dsearch_budget=20', '-Dshow_progress=False'],
            cwd='C:/Users/Shayan Z/Downloads/Tutorial_Stack/Tutorial_Stack')
        process.wait()
        # process = subprocess.Popen(
        #     ['java', '-jar', 'C:/Users/Shayan Z/Downloads/SF100-EvoSuite-20120316/1_tullibee/evosuite-1.0.6.jar',
        #      '-target', 'tullibee.jar',
        #      '-criterion', 'branch', '-Dcrossover_rate={}'.format(crossover), '-Dpopulation={}'.format(population),
        #      '-Dsearch_budget=20', '-Doutput_variables=BranchCoverage', '-Dshow_progress=False'],
        #     cwd='C:/Users/Shayan Z/Downloads/SF100-EvoSuite-20120316/1_tullibee')
        file = pd.read_csv(
            'C:/Users/Shayan Z/Downloads/Tutorial_Stack/Tutorial_Stack/evosuite-report/statistics.csv')
        r, c= file.shape
        total_array = file.at[r-1, 'Total_Goals']
        print(total_array)
        covered_array = file.at[r-1,'Covered_Goals']
        coverage = covered_array / total_array
        self.coverage_dict[state] = coverage
        return coverage


    def step(self, a):
        assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a))

        position = self.state
        row, col = self.to_coordinate(position)
        coverage = self.coverage_dict[position]
        new_row, new_col = self.inc(row,col,a)
        new_position = self.to_position(new_row, new_col)
        new_coverage = self.coverage_dict[new_position]
        done = bool(new_coverage >= self.goal_coverage)
        reward = new_coverage - coverage

        self.state = new_position
        return self.state, reward, done, {}

    def good_available_actions(self, state):
        actions =[]
        for action in range(self.action_space.n):
            position = state
            row, col = self.to_coordinate(position)
            if position in self.coverage_dict:
                coverage = self.coverage_dict[position]
            else:
                coverage = self.b_coverage(position)
            new_row, new_col = self.inc(row, col, action)
            new_position = self.to_position(new_row, new_col)
            if new_position in self.coverage_dict:
                new_coverage = self.coverage_dict[new_position]
            else:
                new_coverage = self.b_coverage(new_position)
            if new_coverage>coverage:
                actions.append(action)
        return actions

    def best_state(self, state):
        actions =self.good_available_actions(state)
        if len(actions)!=0:
            for e in range(len(actions)):
                s, r, done, _ = self.step(e)
                self.best_state(s)
        else:
            print('best position in this grid is:', state)
        return 0