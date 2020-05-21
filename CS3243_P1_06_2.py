# CS3243 Introduction to Artificial Intelligence
# Project 1: k-Puzzle

import os
import sys
import heapq
from collections import deque

# Running script on your own - given code can be run with the command:
# python file.py, ./path/to/init_state.txt ./output/output.txt

# Helper class to contain state information and position of blank tile, etc.
# Contains comparator for PriorityQueue


class Puzzle(object):
    def __init__(self, init_state, goal_state):
        # you may add more attributes if you think is useful
        self.init_state = init_state
        self.goal_state = goal_state
        self.actions = {(0, -1): "RIGHT",
                        (0, 1): "LEFT",
                        (1, 0): "UP",
                        (-1, 0): "DOWN"}
        self.n = len(init_state)
        self.visited = set()
        self.mapping = dict()
        for i, row in enumerate(self.goal_state):
            for j, v in enumerate(row):
                self.mapping[v] = (i, j)

    def solve(self):
        # implement your search algorithm here
        pq = []
        pos = 0
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                if v == 0:
                    pos = (i, j)
        # Node (f(n), path cost, state, 0-position, parent, previous move)
        heapq.heappush(pq, (0, 0, self.init_state, pos, None, (0, 0)))

        while pq:
            curr = heapq.heappop(pq)
            _, p_cost, state, pos, _, p_move = curr
            self.visited.add(tuple(map(tuple, state)))
            # self.visited.add(str(state))
            if self.goal_test(state):
                return self.solution(curr)
            for move in self.actions:
                if move != self.inverse(p_move):
                    dx, dy = move
                    x, y = pos
                    nx = x + dx
                    ny = y + dy
                    if self.is_valid(nx, ny):
                        # new_state = copy.deepcopy(curr.state)
                        new_state = [[v for v in row] for row in state]
                        new_state[x][y] = new_state[nx][ny]
                        new_state[nx][ny] = 0
                        if tuple(map(tuple, new_state)) in self.visited:
                            continue
                        new_node = (self.cost(p_cost, new_state), p_cost + 1,
                                    new_state, (nx, ny), curr, move)
                        heapq.heappush(pq, new_node)

        return ["UNSOLVABLE"]

    # you may add more functions if you think is useful
    def is_valid(self, nx, ny):
        return 0 <= nx < self.n and 0 <= ny < self.n

    def inverse(self, move):
        return tuple([-v for v in move])

    def cost(self, path_cost, curr_state):
        return path_cost + 1 + self.manhattan(curr_state)

    def manhattan(self, state):
        sum = 0
        for i, row in enumerate(state):
            for j, v in enumerate(row):
                if v != 0:
                    x, y = self.mapping[v]
                    sum += abs(i - x) + abs(j - y)
        return sum

    def goal_test(self, state):
        return state == self.goal_state

    def solution(self, node):
        soln = deque()
        while node[4] is not None:
            soln.appendleft(self.actions[node[5]])
            node = node[4]
        return list(soln)

    # adapted from https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html

    def is_solvable(self):
        lst = []
        zeroRow = -1
        for i, row in enumerate(self.init_state):
            # for j, v in enumerate(row):
            for v in row:
                lst.append(v)
                if v == 0:
                    zeroRow = i
        inv = 0
        for i, t in enumerate(lst):
            for v in lst[i+1:]:
                if v != 0 and t != 0 and v < t:
                    inv += 1
        width = len(self.init_state)
        return (width % 2 == 1 and inv % 2 == 0) or (width % 2 == 0 and
                                                     (((self.n - zeroRow + 1) % 2 == 1) == (inv % 2 == 0)))


if __name__ == "__main__":
    # do NOT modify below

    # argv[0] represents the name of the file that is being executed
    # argv[1] represents name of input file
    # argv[2] represents name of destination output file
    if len(sys.argv) != 3:
        raise ValueError("Wrong number of arguments!")

    try:
        f = open(sys.argv[1], 'r')
    except IOError:
        raise IOError("Input file not found!")

    lines = f.readlines()

    # n = num rows in input file
    n = len(lines)
    # max_num = n to the power of 2 - 1
    max_num = n ** 2 - 1

    # Instantiate a 2D list of size n x n
    init_state = [[0 for i in range(n)] for j in range(n)]
    goal_state = [[0 for i in range(n)] for j in range(n)]
    i, j = 0, 0
    for line in lines:
        for number in line.split(" "):
            if number == '':
                continue
            value = int(number, base=10)
            if 0 <= value <= max_num:
                init_state[i][j] = value
                j += 1
                if j == n:
                    i += 1
                    j = 0

    for i in range(1, max_num + 1):
        goal_state[(i-1)//n][(i-1) % n] = i
    goal_state[n - 1][n - 1] = 0

    puzzle = Puzzle(init_state, goal_state)
    ans = puzzle.solve()

    with open(sys.argv[2], 'w') as f:
        for answer in ans:
            f.write(answer+'\n')
