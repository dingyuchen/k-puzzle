# CS3243 Introduction to Artificial Intelligence
# Project 1: k-Puzzle

import os
import sys
import collections
import functools
import copy

# Running script on your own - given code can be run with the command:
# python file.py, ./path/to/init_state.txt ./output/output.txt

class Node(object):
    def __init__(self, cost, state, pos, prev, move):
        self.cost = cost
        self.state = state
        self.pos = pos
        self.prev = prev
        self.move = move


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
        self.pos = (0, 0)

    def solve(self):
        # implement your search algorithm here
        if not self.is_solvable():
            return ["UNSOLVABLE"]
        q = collections.deque()
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                if v == 0:
                    self.pos = (i, j)
        q.append(Node(0, self.init_state, self.pos, None, (0, 0)))

        while q:
            curr = q.popleft()
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
                    self.visited.add(str(new_state))
                    q.append(Node(self.cost(curr, new_state), new_state, (nx, ny), curr, move))

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

    # adapted from https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    def is_solvable(self):
        lst = []
        zeroRow = -1
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                lst.append(v)
                if v == 0: 
                    zeroRow = i
        inv = 0
        for i, t in enumerate(lst):
            for j, v in enumerate(lst[i+1:]):
                if v != 0 and t != 0 and v < t:
                    inv += 1
        width = len(self.init_state)
        return (width % 2 == 1 and inv % 2 == 0) or (width % 2 == 0 and 
                (((self.n - zeroRow + 1) % 2 == 1) == (inv % 2 == 0)))


class Puzzle(Base):
    def cost(self, prev_node, curr_state):
        return prev_node.cost + 1


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
