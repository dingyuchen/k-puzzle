# CS3243 Introduction to Artificial Intelligence
# Project 1: k-Puzzle

import os
import sys
import heapq
import functools
import copy

# Running script on your own - given code can be run with the command:
# python file.py, ./path/to/init_state.txt ./output/output.txt

# Helper class to contain state information and position of blank tile, etc.
# Contains comparator for PriorityQueue
@functools.total_ordering
class Node(object):
    def __init__(self, cost, state, pos, prev, move):
        self.cost = cost
        self.state = state
        self.pos = pos
        self.prev = prev
        self.move = move

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

class Base(object):
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

    def solve(self):
        # implement your search algorithm here
        pq = []
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                if v == 0:
                    self.pos = (i, j)
        heapq.heappush(pq, Node(0, self.init_state, self.pos, None, (0, 0)))

        while pq:
            curr = heapq.heappop(pq)
            self.visited.add(str(curr.state))
            if self.goal_test(curr.state):
                return self.solution(curr)
            for move in self.actions:
                dx, dy = move
                x, y = curr.pos
                nx = x + dx
                ny = y + dy
                if self.is_valid(nx, ny):
                    new_state = copy.deepcopy(curr.state)
                    new_state[x][y] = new_state[nx][ny] 
                    new_state[nx][ny] = 0
                    if str(new_state) in self.visited:
                        continue
                    new_node = Node(self.cost(curr, new_state), new_state, (nx, ny), curr, move)
                    heapq.heappush(pq, new_node)

        return ["UNSOLVABLE"] 

    # you may add more functions if you think is useful
    def is_valid(self, nx, ny):
        return nx >= 0 and nx < self.n and ny < self.n and ny >= 0

    def cost(self, prev_node, curr_state):
        return prev_node.cost + 1

    def goal_test(self, state):
        for i, row in enumerate(state):
            for j, v in enumerate(row):
                if v != self.goal_state[i][j]:
                    return False
        return True

    def solution(self, node):
        soln = []
        curr = node
        while curr.prev is not None:
            soln.append(self.actions[curr.move])
            curr = curr.prev
        return soln[::-1]


# Wrapper class to inject cost function
class Puzzle(Base):
    def __init__(self, init_state, goal_state):
        super(Puzzle, self).__init__(init_state, goal_state)
        self.mapping = dict()
        for i, row in enumerate(self.goal_state):
            for j, v in enumerate(row):
                self.mapping[v] = (i, j)

    def cost(self, prev_node, curr_state):
        return prev_node.cost + 1 + self.manhattan(curr_state)

    def manhattan(self, state):
        sum = 0
        for i, row in enumerate(state):
            for j, v in enumerate(row):
                x, y = self.mapping[v]
                sum += abs(i - x) + abs(j - y)
        return sum



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


    i,j = 0, 0
    for line in lines:
        for number in line.split(" "):
            if number == '':
                continue
            value = int(number , base = 10)
            if  0 <= value <= max_num:
                init_state[i][j] = value
                j += 1
                if j == n:
                    i += 1
                    j = 0

    for i in range(1, max_num + 1):
        goal_state[(i-1)//n][(i-1)%n] = i
    goal_state[n - 1][n - 1] = 0

    puzzle = Puzzle(init_state, goal_state)
    ans = puzzle.solve()

    with open(sys.argv[2], 'a') as f:
        for answer in ans:
            f.write(answer+'\n')
