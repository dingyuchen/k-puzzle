# CS3243 Introduction to Artificial Intelligence
# Project 1: k-Puzzle

import os
import sys
try:
   import queue # python 3
except ImportError:
   import Queue as queue # python 2
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
        if not self.isSolvable():
            return ["UNSOLVABLE"]
        q = queue.Queue();
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                if v == 0:
                    self.pos = (i, j)
        q.put(Node(0, self.init_state, self.pos, None, (0, 0)))

        while q:
            curr = q.get()
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
                    new_node = Node(self.cost(curr, new_state), new_state, (nx, ny), curr, move)
                    q.put(new_node)

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


    def isSolvable(self):
        lst = []
        zeroRow = -1
        for row in range(len(self.init_state)):
            for j in range(len(self.init_state[0])):
                lst.append(self.init_state[row][j])
                if j == 0:
                    zeroRow = row
        inv = 0
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if lst[j] != 0 and lst[i] != 0 and lst[j] < lst[i]:
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
