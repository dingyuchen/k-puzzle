# CS3243 Introduction to Artificial Intelligence
# Project 1: k-Puzzle

import os
import sys
import heapq
from collections import deque

# Running script on your own - given code can be run with the command:
# python file.py, ./path/to/init_state.txt ./output/output.txt


class Node(object):
    """
    Helper class to contain various information of node, 
    including breakdown of heuristic calculation to reduce recomputation 
    """

    def __init__(self, state, zero_pos, hmap, immut=None, path_cost=0, parent=None, prev_move=(0, 0)):
        self.path_cost = path_cost
        self.state = state
        self.zero_pos = zero_pos
        self.parent = parent
        self.prev_move = prev_move
        self.immut = immut or tuple(map(tuple, state))
        self.hmap = hmap


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
        self.pos = (-1, -1)
        self.mapping = dict()
        for i, row in enumerate(self.goal_state):
            for j, v in enumerate(row):
                self.mapping[v] = (i, j)

    def solve(self):
        # implement your search algorithm here
        if not self.is_solvable():
            return ["UNSOLVABLE"]
        pq = []
        assert self.pos[0] >= 0 and self.pos[1] >= 0
        # save conflict layouts to prevent recomputation
        row_conflicts = [self.line_conflict(i, row, lambda v: self.mapping[v])
                         for i, row in enumerate(self.init_state)]
        col_conflicts = [self.line_conflict(i, row, lambda v: self.mapping[v][::-1])
                         for i, row in enumerate(self.transpose(self.init_state))]
        md = self.manhattan(self.init_state)
        hmap = {"row": row_conflicts, "col": col_conflicts, "md": md}
        node = Node(self.init_state, self.pos, hmap)
        heapq.heappush(pq, (0, node))

        while pq:
            _, curr = heapq.heappop(pq)
            self.visited.add(curr.immut)
            if self.goal_test(curr.state):
                return self.solution(curr)
            for move in self.actions:
                if move == self.undo(curr.prev_move):
                    continue

                dx, dy = move
                x, y = curr.zero_pos
                nx, ny = x + dx, y + dy

                if self.is_valid(nx, ny):
                    new_state = [[v for v in row] for row in curr.state]
                    new_state[x][y] = new_state[nx][ny]
                    new_state[nx][ny] = 0
                    state_hash = tuple(map(tuple, new_state))

                    if state_hash in self.visited:
                        continue

                    new_hmap = self.update_hmap(curr, move, new_state)
                    new_node = Node(new_state, (nx, ny), new_hmap,
                                    state_hash, curr.path_cost + 1, curr, move)
                    heapq.heappush(pq, (self.cost(new_node), new_node))

        return ["UNSOLVABLE"]

    # you may add more functions if you think is useful
    def is_valid(self, nx, ny):
        return 0 <= nx < self.n and 0 <= ny < self.n

    def undo(self, move):
        return tuple([-v for v in move])

    def cost(self, node):
        lc = sum(node.hmap["row"]) + sum(node.hmap["col"])
        return node.path_cost + 1 + node.hmap["md"] + 2 * lc

    def update_hmap(self, parent, move, child_state):
        dx, dy = move
        x, y = parent.zero_pos
        nx, ny = x + dx, y + dy
        v = parent.state[nx][ny]

        new_md = parent.hmap["md"] - self.single_manhattan(
            nx, ny, v) + self.single_manhattan(x, y, v)

        new_hmap = {k: v for k, v in parent.hmap.items()}
        new_hmap["md"] = new_md
        if dx == 0:
            # update col heuristics
            col_conflicts = [v for v in parent.hmap["col"]]
            col_conflicts[y] = self.line_conflict(
                y, [row[y] for row in child_state], self.col_dest)
            col_conflicts[ny] = self.line_conflict(
                ny, [row[ny] for row in child_state], self.col_dest)
            new_hmap["col"] = col_conflicts
        else:
            # update row heuristics
            row_conflicts = [v for v in parent.hmap["row"]]
            row_conflicts[x] = self.line_conflict(
                x, child_state[x], self.row_dest)
            row_conflicts[nx] = self.line_conflict(
                nx, child_state[nx], self.row_dest)
            new_hmap["row"] = row_conflicts
        return new_hmap

    def row_dest(self, v):
        return self.mapping[v]

    def col_dest(self, v):
        return self.row_dest(v)[::-1]

    def single_manhattan(self, i, j, v):
        assert v != 0
        x, y = self.mapping[v]
        return abs(i - x) + abs(j - y)

    def manhattan(self, state):
        sum = 0
        for i, row in enumerate(state):
            for j, v in enumerate(row):
                if v != 0:
                    sum += self.single_manhattan(i, j, v)
        return sum

    @staticmethod
    def transpose(state):
        width = len(state[0])
        return [[state[j][i]
                 for j in range(width)] for i in range(len(state))]

    # Original paper found here:
    # https://cse.sc.edu/~mgv/csce580sp15/gradPres/HanssonMayerYung1992.pdf
    def line_conflict(self, i, line, dest):
        """
        If 2 tiles are both in their respective goal rows/columns but eventually require to swap places with one another to reach their goal positions, then these 2 tiles are said to be in linear conflict.
        We know that 1 tile must definitely move out of the row in order to let the other tile pass, which adds 2 extra moves to the manhattan distance heuristic, while still being admissble.
        """
        conflicts = 0
        conflict_graph = {}
        for j, u in enumerate(line):
            if u == 0:
                continue
            x, y = dest(u)
            if i != x:
                continue

            for k in range(j + 1, self.n):
                # opposing tile
                v = line[k]
                if v == 0:
                    continue
                # print(i, "comparing ", u, v)
                tx, ty = dest(v)
                if tx == x and ty <= y:
                    u_degree, u_nbrs = conflict_graph.get(u) or (0, set())
                    u_nbrs.add(v)
                    conflict_graph[u] = (u_degree + 1, u_nbrs)
                    v_degree, v_nbrs = conflict_graph.get(v) or (0, set())
                    v_nbrs.add(u)
                    conflict_graph[v] = (v_degree + 1, v_nbrs)
        # print("conflict graph", conflict_graph)
        while sum([v[0] for v in conflict_graph.values()]) > 0:
            # resolve most conflicting tile first
            popped = max(conflict_graph.keys(),
                         key=lambda k: conflict_graph[k][0])
            for neighbour in conflict_graph[popped][1]:
                degree, vs = conflict_graph[neighbour]
                vs.remove(popped)
                conflict_graph[neighbour] = (degree - 1, vs)
                conflicts += 1
            conflict_graph.pop(popped)
        return conflicts

    def goal_test(self, state):
        return state == self.goal_state

    def solution(self, node):
        soln = deque()
        while node.parent is not None:
            soln.appendleft(self.actions[node.prev_move])
            node = node.parent
        return list(soln)

    # adapted from https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    def is_solvable(self):
        lst = []
        zeroRow = -1
        for i, row in enumerate(self.init_state):
            for j, v in enumerate(row):
                lst.append(v)
                if v == 0:
                    zeroRow = i
                    self.pos = (i, j)
        inv = 0
        for i, t in enumerate(lst):
            for v in lst[i+1:]:
                if v != 0 and t != 0 and v < t:
                    inv += 1
        width = len(self.init_state)
        return (width % 2 == 1 and inv % 2 == 0) or (width % 2 == 0 and
                                                     (((self.n - zeroRow) % 2 == 1) == (inv % 2 == 0)))


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
