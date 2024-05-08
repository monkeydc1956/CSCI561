import copy
from copy import deepcopy
import math
import numpy as np
from read import *
from write import *
# move by dfs

class MyPlayer:
    def __init__(self, piece_type, pre_board, cur_board):
        self.name = "minimax_pruning"
        self.piece_type = piece_type
        self.pre_board = pre_board
        self.cur_board = cur_board
        self.N = 5
        self.maxValue = math.inf
        self.minValue = -math.inf
        self.max_step = 24
        self.max_depth = 4
        self.next_steps = [[1,0], [-1,0], [0,1], [0,-1]]
        self.candy_size = 18 # branching factoe
        self.reward = []
        # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
        # self.reward.append((-self.reward_base, 0, self.reward_base, 0, -self.reward_base))
        # self.reward.append((0, self.reward_base, self.reward * 2, self.reward_base, 0))
        # self.reward.append((-self.reward_base, 0, self.reward_base, 0, -self.reward_base))
        # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
        self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
        self.reward.append((-self.reward_base, 5, self.reward_base, 5, -self.reward_base))
        self.reward.append((0, self.reward_base, self.reward * 2, self.reward_base, 0))
        self.reward.append((-self.reward_base, 5, self.reward_base, 5, -self.reward_base))
        self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
        # -100 -50 0 -50 -100
        # -50 0 50 0 -50
        # 0 50 100 50 0
        # -50 0 50 0 -50
        # -100 -50 0 -50 -100
        # 1 = white | 2 = black

    def minimax(self):
        cur_step = self.get_steps()
        move, _ = self.maxVal(self.cur_board, self.piece_type, 0,self.minValue, self.maxValue, None, cur_step, False)
        if move is None or move == "STOP":
            return "PASS"
        else:
            return move

    def maxVal(self, cur_board, piece_type, current_depth, alpha, beta, last_move, cur_step, skip_end):
        if self.max_depth == current_depth or cur_step + current_depth == self.max_step:
            return self.evaluate(cur_board, piece_type)
        if skip_end:
            return self.evaluate(cur_board, piece_type)
        skip_end = False
        res_max = self.minValue
        max_move = None
        candidates = self.get_candidats(cur_board, piece_type, self.candy_size)
        candidates.append("STOP")
        if last_move == "STOP":
            skip_end = True
        for candidate in candidates:
            if candidate == "STOP":
                next_board = copy.deepcopy(cur_board)
            else:
                next_board = self.move(cur_board, piece_type, candidate)
            res_min = self.minVal(next_board, 3 - piece_type, current_depth + 1, alpha, beta, candidate, cur_step, skip_end)
            if res_max < res_min:
                res_max = res_min
                max_move = candidate
            if res_max >= beta:
                if current_depth == 0:
                    return max_move, res_max
                else:
                    return res_max
            alpha = max(alpha, res_max)
        if current_depth == 0:
            return max_move, res_max
        else:
            return res_max

    def minVal(self, cur_board, piece_type, current_depth, alpha, beta, last_move, cur_step, skip_end):
        if self.max_depth == current_depth:
            return self.evaluate(cur_board, piece_type)
        if cur_step + current_depth == self.max_step or skip_end:
            return self.evaluate(cur_board, self.piece_type)
        skip_end = False
        res_min = self.maxValue
        candidates = self.get_candidats(cur_board, piece_type, self.candy_size)
        candidates.append("STOP")
        if last_move == "STOP":
            skip_end = True
        for candidate in candidates:
            if candidate == "STOP":
                next_board = copy.deepcopy(cur_board)
            else:
                next_board = self.move(cur_board, piece_type, candidate)
            res_max = self.maxVal(next_board, 3 - piece_type, current_depth + 1, alpha, beta, candidate, cur_step, skip_end)
            if res_max < res_min:
                res_min = res_max
            if res_min <= alpha:
                return res_min
            beta = min(beta, res_min)
        return res_min

    def evaluate(self, cur_board, piece_type):
        bonus_factor = 8
        pos_count = 0
        pos_liberty = set()
        neg_count= 0
        neg_liberty = set()
        for i in range(self.N):
            for j in range(self.N):
                if cur_board[i][j] == piece_type:
                    pos_count += 1
                elif cur_board[i][j] == 3 - piece_type:
                    neg_count+= 1
                else:
                    for index in range(len(self.next_steps)):
                        tmp_i = i + self.next_steps[index][0]
                        tmp_j = j + self.next_steps[index][1]
                        if self.validationPos(tmp_i, tmp_j):
                            if cur_board[tmp_i][tmp_j] == piece_type:
                                pos_liberty.add((i, j))
                            elif cur_board[tmp_i][tmp_j] == 3 - piece_type:
                                neg_liberty.add((i, j))
        edge = 0
        for j in range(self.N):
            if cur_board[0][j] == piece_type or cur_board[self.N - 1][j] == piece_type:
                edge += 1
        for j in range(1, self.N - 1):
            if cur_board[j][0] == piece_type or cur_board[j][self.N - 1] == piece_type:
                edge += 1
        free = 0
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if cur_board[i][j] == 0:
                    free += 1
        score = (min(max((len(pos_liberty) - len(neg_liberty)), -bonus_factor), bonus_factor)
                 + (-bonus_factor * self.calculate_euler_number(cur_board, piece_type))/2
                 + ((bonus_factor-3) * (pos_count - neg_count))
                 - ((bonus_factor+1) * edge * (free / (bonus_factor+1))))
        if self.piece_type == 2:
            score += 2.5

        return score

    def validationPos(self,i,j):
        if 0 <= i < self.N and 0 <= j < self.N:
            return True
        else:
            return  False

    import copy

    def move(self, cur_board, piece_type, move):
        next_board = copy.deepcopy(cur_board)
        i, j = move
        next_board[i][j] = piece_type
        deleted = False
        def dfs(i, j):
            nonlocal deleted
            if not self.validationPos(i, j):
                return
            if next_board[i][j] == 0:
                deleted = False
                return
            if next_board[i][j] == 3 - piece_type:
                next_board[i][j] = 0
                for x, y in self.next_steps:
                    dfs(i + x, j + y)
        for x, y in self.next_steps:
            new_i, new_j = i + x, j + y
            if self.validationPos(new_i, new_j) and next_board[new_i][new_j] == 3 - piece_type:
                deleted = True
                dfs(new_i, new_j)

        return next_board

    def calculate_euler_number(self, cur_board, piece_type):
        windows = 2
        next_board = np.zeros((self.N + windows, self.N + windows), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                next_board[i + 1][j + 1] = cur_board[i][j]
        pos_diag, pos_neg_diag, pos_squ = 0, 0, 0
        neg_diag, neg_neg_diag, neg_squ = 0, 0, 0
        for i in range(self.N):
            for j in range(self.N):
                mini_board = next_board[i: i + windows, j: j + windows]
                pos_diag += self.diag_score(mini_board, piece_type)
                pos_neg_diag += self.neg_diag(mini_board, piece_type)
                pos_squ += self.square_score(mini_board, piece_type)
                neg_diag += self.diag_score(mini_board, 3 - piece_type)
                neg_neg_diag += self.neg_diag(mini_board, 3 - piece_type)
                neg_squ += self.square_score(mini_board, 3 - piece_type)
        return ((pos_diag+pos_neg_diag*windows-pos_squ)-(neg_diag+neg_neg_diag*windows-neg_squ)) / windows*2
    def get_candidats(self, cur_board, piece_type, candy_size):
        cand_taken = []
        cand_kill = []
        cand_defend = []
        for i in range(self.N):
            for j in range(self.N):
                if cur_board[i][j] == 0:
                    if self.liberty(cur_board, i, j, piece_type):
                        if not self.check_for_ko(i, j):
                            if i == 0 or i == self.N - 1 or j == 0  or j == self.N - 1:
                                cand_taken.append((i, j))
                            else:
                                cand_kill.append((i, j))
                    else:
                        for index in range(len(self.next_steps)):
                            tmp_i = i + self.next_steps[index][0]
                            tmp_j = j + self.next_steps[index][1]
                            if self.validationPos(tmp_i, tmp_j):
                                if cur_board[tmp_i][tmp_j] == 3 - piece_type:
                                    next_board = copy.deepcopy(cur_board)
                                    next_board[i][j] = piece_type
                                    if not self.liberty(next_board, tmp_i, tmp_j,3 - piece_type):
                                        if not self.check_for_ko(i, j):
                                            cand_defend.append((i, j))
                                        break
        candidates_list = cand_kill+cand_defend+cand_taken
        return candidates_list[:candy_size]

    def liberty(self, cur_board, i, j, piece_type):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            for index in range(len(self.next_steps)):
                tmp_i = top_node[0] + self.next_steps[index][0]
                tmp_j = top_node[1] + self.next_steps[index][1]
                if self.validationPos(tmp_i, tmp_j):
                    if (tmp_i, tmp_j) in visited:
                        continue
                    elif cur_board[tmp_i][tmp_j] == 0:
                        return True
                    elif cur_board[tmp_i][tmp_j] == piece_type and (tmp_i, tmp_j) not in visited:
                        stack.append((tmp_i, tmp_j))
        return False


    def check_for_ko(self, i, j):
        if self.pre_board[i][j] != self.piece_type:
            return False
        next_board = copy.deepcopy(self.cur_board)
        next_board[i][j] = self.piece_type
        opponent_i, opponent_j = self.opponent_move()
        for index in range(len(self.next_steps)):
            tmp_i = i + self.next_steps[index][0]
            tmp_j = j + self.next_steps[index][1]
            if tmp_i == opponent_i and tmp_j == opponent_j:
                if not self.liberty(next_board, tmp_i, tmp_j, 3 - self.piece_type):
                    self.delete_group(next_board, tmp_i, tmp_j, 3 - self.piece_type)
        return np.array_equal(next_board, self.pre_board)

    def opponent_move(self):
        if np.array_equal(self.cur_board, self.pre_board):
            return None
        for i in range(self.N):
            for j in range(self.N):
                if self.cur_board[i][j] != self.pre_board[i][j] \
                        and self.cur_board[i][j] != 0:
                    return i, j

    def delete_group(self, cur_board, i, j, piece_type):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            cur_board[top_node[0]][top_node[1]] = 0
            for index in range(len(self.next_steps)):
                tmp_i = top_node[0] + self.next_steps[index][0]
                tmp_j = top_node[1] + self.next_steps[index][1]
                if self.validationPos(tmp_i, tmp_j):
                    if (tmp_i, tmp_j) in visited:
                        continue
                    elif cur_board[tmp_i][tmp_j] == piece_type:
                        stack.append((tmp_i, tmp_j))
        return cur_board


    def diag_score(self, mini_board: tuple, piece_type: int) -> int:
        top_left = mini_board[0][0]
        top_right = mini_board[0][1]
        bottom_left = mini_board[1][0]
        bottom_right = mini_board[1][1]
        if (top_left == piece_type and top_right != piece_type and bottom_left != piece_type and bottom_right != piece_type) or \
                (top_left != piece_type and top_right == piece_type and bottom_left != piece_type and bottom_right != piece_type) or \
                (top_left != piece_type and top_right != piece_type and bottom_left == piece_type and bottom_right != piece_type) or \
                (top_left != piece_type and top_right != piece_type and bottom_left != piece_type and bottom_right == piece_type):
            return 1
        else:
            return 0

    def neg_diag(self, mini_board, piece_type):
        top_left = mini_board[0][0]
        top_right = mini_board[0][1]
        bottom_left = mini_board[1][0]
        bottom_right = mini_board[1][1]
        condition1 = (top_left == piece_type and top_right != piece_type and bottom_left != piece_type and bottom_right == piece_type)
        condition2 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right != piece_type)
        return 1 if condition1 or condition2 else 0

    def square_score(self, mini_board, piece_type):
        top_left = mini_board[0][0]
        top_right = mini_board[0][1]
        bottom_left = mini_board[1][0]
        bottom_right = mini_board[1][1]
        condition1 = (top_left == piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right != piece_type)
        condition2 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right == piece_type)
        condition3 = (top_left == piece_type and top_right != piece_type and bottom_left == piece_type and bottom_right == piece_type)
        condition4 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right == piece_type)
        return 1 if condition1 or condition2 or condition3 or condition4 else 0

    def edge_case(self, cur_board, piece_type):
        pos_edge_count = 0
        for j in range(self.N):
            if cur_board[0][j] == piece_type or cur_board[self.N - 1][j] == piece_type:
                pos_edge_count += 1

        for j in range(1, self.N - 1):
            if cur_board[j][0] == piece_type or cur_board[j][self.N - 1] == piece_type:
                pos_edge_count += 1

        return pos_edge_count

    def get_steps(self):
        pre_board_init = True
        cur_board_init = True
        for i in range(4):
            if sum(self.pre_board[i]) > 0:
                pre_board_init = False
                cur_board_init = False
                break
            if sum(self.cur_board[i]) > 0:
                cur_board_init = False
        if pre_board_init and cur_board_init:
            stones = 0
        elif pre_board_init and not cur_board_init:
            stones = 1
        else:
            with open('step_test.txt') as f:
                stones = int(f.readline())
                stones += 2
        with open('step_test.txt', 'w') as f:
            f.write(f'{stones}')
        return stones



if __name__ == '__main__':
    N = 5
    piece_type, pre_board, cur_board = readInput(N)
    my_player = MyPlayer(piece_type, pre_board, cur_board)
    action = my_player.minimax()
    writeOutput(action)
