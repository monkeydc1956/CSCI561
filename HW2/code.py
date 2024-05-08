from copy import deepcopy
import math
import numpy as np
# from read import *
# from write import *
# minVal improved, bettersore, hueristic tianyuan hueristic

class GoPlayer:
    def __init__(self, piece_type, pre_board, cur_board):
        self.name = "minimax_pruning"
        self.piece_type = piece_type
        self.pre_board = pre_board
        self.cur_board = cur_board
        self.N = 5
        self.maxValue = math.inf
        self.minValue = -math.inf
        self.reward = []
        self.reward_base = 3
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
        self.max_step = 24
        self.max_depth = 4
        self.next_steps = [[1,0], [-1,0], [0,1], [0,-1]]
        self.candy_size = 20 # branching factoe
        # 2 = white | 1 = black

    def minimax_pruning_with_heuristic(self):
        cur_step = self.get_steps()
        if cur_step == 0: # black
            return (2, 2)
        if cur_step == 1: # white
            if self.cur_board[2][2] == 0:
                return (2, 2) # central advantage
            else:
                if self.cur_board[1][1] == 0:
                    return (1, 1)
        # special step four -- 'tianyuan' start # white
        if cur_step == 3:
            if self.cur_board[2][2] != 0:
                if self.cur_board[1][2] != 0 and self.cur_board[2][1] == 0:
                    return (2, 1)
                elif self.cur_board[2][1] != 0 and self.cur_board[1][2] == 0:
                    return (1, 2)
                elif self.cur_board[3][2] != 0 and self.cur_board[1][2] == 0:
                    return (1, 2)
                elif self.cur_board[2][3] != 0 and self.cur_board[2][1] == 0:
                    return (2, 1)
        # aggressive arrange
        move, _ = self.maxVal(None, cur_step, 0, self.cur_board, self.piece_type, 0,self.minValue, self.maxValue)
        if move is None or move == "STOP":
            return "PASS"
        else:
            return move

    def minVal(self, last_move, cur_step, skip_end, cur_board, piece_type, cur_depth, alpha, beta):
        if self.max_depth == cur_depth:
            return self.evaluate(cur_board, piece_type)
        if cur_step + cur_depth == self.max_step or skip_end:
            return self.evaluate(cur_board, self.piece_type)
        skip_end = 0
        res_min = self.maxValue
        min_move = None
        candidates = self.get_candidats(cur_board, piece_type)
        candidates.append("STOP")
        if last_move == "STOP":
            skip_end = 1
        for candidate in candidates:
            if candidate == "STOP":
                next_board = deepcopy(cur_board)
            else:
                next_board = self.take_action(cur_board, piece_type, candidate)
            res_max = self.maxVal(candidate, cur_step, skip_end, next_board, 3 - piece_type, cur_depth + 1, alpha, beta)
            if res_max < res_min:
                res_min = res_max
                min_move = candidate
            if res_min <= alpha:
                if cur_depth > 0:
                    return res_min
                else:
                    return min_move, res_min
            if res_min < beta:
                beta = res_min
        if cur_depth == 0:
            return min_move, res_min
        else:
            return res_min

    def maxVal(self, last_move, cur_step, skip_end, cur_board, piece_type, cur_depth, alpha, beta):
        if self.max_depth == cur_depth or cur_step + cur_depth == self.max_step:
            return self.evaluate(cur_board, piece_type)
        if skip_end == 1:
            return self.evaluate(cur_board, piece_type)
        skip_end = 0
        res_max = self.minValue
        max_move = None
        candidates = self.get_candidats(cur_board, piece_type)
        candidates.append("STOP")
        if last_move == "STOP":
            skip_end = 1
        for candidate in candidates:
            if candidate == "STOP":
                next_board = deepcopy(cur_board)
            else:
                next_board = self.take_action(cur_board, piece_type, candidate)
            res_min = self.minVal(candidate, cur_step, skip_end, next_board, 3 - piece_type, cur_depth + 1, alpha, beta)
            if res_max < res_min:
                res_max = res_min
                max_move = candidate
            if res_max >= beta:
                if cur_depth > 0:
                    return res_max
                else:
                    return max_move, res_max
            if res_max>alpha:
                alpha = res_max
        if cur_depth == 0: # reserved for first layer
            return max_move, res_max
        else:
            return res_max

    def evaluate(self, cur_board, piece_type):
        edge = 0
        for j in range(self.N):
            if cur_board[0][j] == piece_type or cur_board[self.N - 1][j] == piece_type:
                edge += 1
        for j in range(1, self.N - 1):
            if cur_board[j][0] == piece_type or cur_board[j][self.N - 1] == piece_type:
                edge += 1
        bonus_factor = 8
        pos_count = 0
        neg_count = 0
        free = 0
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if cur_board[i][j] == 0:
                    free += 1
        pos_liberty = set()
        neg_liberty = set()
        mid_liberty = set()
        for i in range(self.N):
            for j in range(self.N):
                if cur_board[i][j] == piece_type:
                    pos_count += 1
                elif cur_board[i][j] == 3 - piece_type:
                    neg_count += 1
                else:
                    for index in range(len(self.next_steps)):
                        tmp_i, tmp_j = i + self.next_steps[index][0], j + self.next_steps[index][1]
                        if self.validationPos(tmp_i, tmp_j) and cur_board[tmp_i][tmp_j] == piece_type:
                                pos_liberty.add((i, j))
                        elif self.validationPos(tmp_i, tmp_j) and cur_board[tmp_i][tmp_j] == 3 - piece_type:
                                neg_liberty.add((i, j))
        count_score = (bonus_factor-3) * (pos_count - neg_count)
        liberty_score = min(max((len(pos_liberty) - len(neg_liberty)), -bonus_factor), bonus_factor)
        position_score = (bonus_factor+1) * edge * (free / (bonus_factor+1))
        loc_score = (-bonus_factor * self.location_evalution(cur_board, piece_type))/2
        final_score = count_score+ loc_score+ liberty_score - position_score
        if self.piece_type == 2:
            return final_score + 2.5
        else:
            return final_score

    def take_action(self, cur_board, piece_type, step):
        next_board = deepcopy(cur_board)
        next_board[step[0]][step[1]] = piece_type

        def dfs(tmp_i, tmp_j, visited):
            visited.add((tmp_i, tmp_j))
            for x, y in self.next_steps:
                new_i, new_j = tmp_i + x, tmp_j + y
                if self.validationPos(new_i, new_j) and (new_i, new_j) not in visited:
                    if next_board[new_i][new_j] == 0:
                        return False
                    elif next_board[new_i][new_j] == 3 - piece_type:
                        if dfs(new_i, new_j, visited) is False:
                            return False
            return True

        for x, y in self.next_steps:
            tmp_i, tmp_j = step[0] + x, step[1] + y
            if self.validationPos(tmp_i, tmp_j) and next_board[tmp_i][tmp_j] == 3 - piece_type:
                visited = set()
                if dfs(tmp_i, tmp_j, visited):
                    for stone in visited:
                        next_board[stone[0]][stone[1]] = 0
        return next_board

    def location_evalution(self, cur_board, piece_type):
        windows = 2
        next_board = np.zeros((self.N + windows, self.N + windows), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                next_board[i + 1][j + 1] = cur_board[i][j]

        pos = self.local_score(next_board, windows, piece_type)
        pos_one_sq, dia_pos, pos_three_sq = pos[0], pos[1], pos[2]
        neg = self.local_score(next_board, windows, 3 - piece_type)
        neg_one_sq, dia_neg, neg_three_sq = neg[0], neg[1], neg[2]
        return ((pos_one_sq+windows*dia_pos-pos_three_sq)-(neg_one_sq+windows*dia_neg-neg_three_sq))/(windows*windows)

    def local_score(self, cur_board, windows, piece_type):
        res = [0,0,0]
        for i in range(self.N):
            for j in range(self.N):
                mini_board = cur_board[i: i + windows, j: j + windows]
                res[0] += self.square_one(mini_board, piece_type)
                res[1] += self.diag_two(mini_board, piece_type)
                res[2] += self.square_three(mini_board, piece_type)
        return res

    def liberty_valid(self, cur_board, i, j, piece_type):
        candy = []
        candy.append((i, j))
        dfs = set()
        while len(candy) > 0:
            tmp = candy.pop()
            dfs.add(tmp)
            for index in range(len(self.next_steps)):
                tmp_i, tmp_j = tmp[0] + self.next_steps[index][0], tmp[1] + self.next_steps[index][1]
                if self.validationPos(tmp_i, tmp_j):
                    if (tmp_i, tmp_j) not in dfs:
                        if cur_board[tmp_i][tmp_j] == 0:
                            return True
                        elif cur_board[tmp_i][tmp_j] == piece_type:
                            if (tmp_i, tmp_j) not in dfs:
                                candy.append((tmp_i, tmp_j))
        return False

    def get_candidats(self, cur_board, piece_type):
        candidates_taken = []
        candidates_kill = []
        candidates_defend = []
        for i in range(self.N):
            for j in range(self.N):
                if cur_board[i][j] == 0:
                    if not self.liberty_valid(cur_board, i, j, piece_type):
                        for index in range(len(self.next_steps)):
                            tmp_i, tmp_j = i + self.next_steps[index][0], j + self.next_steps[index][1]
                            if self.validationPos(tmp_i, tmp_j):
                                if cur_board[tmp_i][tmp_j] == 3 - piece_type:
                                    next_board = deepcopy(cur_board)
                                    next_board[i][j] = piece_type
                                    if not self.liberty_valid(next_board, tmp_i, tmp_j,3 - piece_type):
                                        if not self.KO_RULES(i, j):
                                            candidates_kill.append((i, j))
                                        break
                    else:
                        if not self.KO_RULES(i, j):
                            if i == 0 or j == 0 or i == self.N - 1 or j == self.N - 1:
                                candidates_taken.append((i, j))
                            else:
                                candidates_defend.append((i, j))
        candidates_list = candidates_kill+candidates_defend+candidates_taken
        return candidates_list[:self.candy_size]

    def KO_RULES(self, i, j):
        if self.pre_board[i][j] != self.piece_type:
            return False
        next_i, next_j = -1, -1
        next_board = deepcopy(self.cur_board)
        next_board[i][j] = self.piece_type
        #next_i, next_j = self.move_next()
        for a in range(self.N):
            for b in range(self.N):
                if self.cur_board[a][b] != self.pre_board[a][b]:
                    if self.cur_board[a][b] != 0:
                        next_i, next_j = a, b
                        break
        for index in range(len(self.next_steps)):
            tmp_i, tmp_j = i + self.next_steps[index][0], j + self.next_steps[index][1]
            if tmp_i == next_i and tmp_j == next_j:
                if not self.liberty_valid(next_board, tmp_i, tmp_j, 3 - self.piece_type):
                    self.update_board(next_board, tmp_i, tmp_j, 3 - self.piece_type)
        return self.compare_board(next_board)

    def compare_board(self, cur_board):
        for i in range(len(cur_board)):
            for j in range(len(cur_board[i])):
                if cur_board[i][j] != self.pre_board[i][j]:
                    return False
        return True

    def move_next(self):
        if self.compare_board(self.cur_board):
            return None
        for i in range(self.N):
            for j in range(self.N):
                if self.cur_board[i][j] != self.pre_board[i][j]:
                    if self.cur_board[i][j] != 0:
                        return i, j

    def validationPos(self,i,j):
        if 0 <= i < self.N and 0 <= j < self.N:
            return True
        else:
            return False

    def update_board(self, cur_board, i, j, piece_type):
        def dfs(tmp_i, tmp_j):
            if not self.validationPos(tmp_i, tmp_j) or cur_board[tmp_i][tmp_j] != piece_type:
                return
            cur_board[tmp_i][tmp_j] = 0
            for x, y in self.next_steps:
                new_i, new_j = tmp_i + x, tmp_j + y
                dfs(new_i, new_j)
        if cur_board[i][j] == piece_type:
            dfs(i, j)
        return cur_board

    def detect_eye(self, piece_type, cur_board, i, j):
        board = cur_board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        enemy = 0
        blank = 0
        for nei in neighbors:
            if cur_board[nei[0]][nei[1]] == 3 - piece_type:
                enemy += 1
            if cur_board[nei[0]][nei[1]] == 0:
                blank += 1
        if enemy == 0 and blank == 0:
            return True
        else:
            return False

    def detect_secondry_eye(self, piece_type, cur_board, i, j):
        board = cur_board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        enemy = 0
        blank = 0
        for nei in neighbors:
            if cur_board[nei[0]][nei[1]] == 3 - piece_type:
                enemy += 1
            if cur_board[nei[0]][nei[1]] == 0:
                blank += 1
        if enemy == 0 and blank <= 2:
            return True
        else:
            return False

    def future_adv(self, i, j, cur_board):
        return self.sur_score(i, j, cur_board) + self.ally_score(i, j, cur_board)

    def sur_score(self, i, j, cur_board):
        score = 0
        for nei in self.detect_neighbor(i, j, cur_board):
            if cur_board[nei[0]][nei[1]] == 0:
                score += 1
        return score

    def square_three(self, mini_board, piece_type):
        top_left = mini_board[0][0]
        top_right = mini_board[0][1]
        bottom_left = mini_board[1][0]
        bottom_right = mini_board[1][1]
        condition1 = (top_left == piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right != piece_type)
        condition2 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right == piece_type)
        condition3 = (top_left == piece_type and top_right != piece_type and bottom_left == piece_type and bottom_right == piece_type)
        condition4 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right == piece_type)
        return 1 if condition1 or condition2 or condition3 or condition4 else 0

    def ally_score(self, i, j, cur_board):
        score = 0
        for friend in self.survive_ways(i, j):
            if cur_board[friend[0]][friend[1]] == self.piece_type:
                score += 1
        return score

    def survive_ways(self, i, j):
        diags = []
        if i - 1 >= 0 and j - 1 >= 0:
            diags.append((i - 1, j - 1))
        if i - 1 >= 0 and j + 1 < self.N:
            diags.append((i - 1, j + 1))
        if i + 1 < self.N and j - 1 >= 0:
            diags.append((i + 1, j - 1))
        if i + 1 < self.N and j + 1 < self.N:
            diags.append((i + 1, j + 1))
        return diags

    def around_enemies(self, i, j, piece_type, cur_board):
        neis = self.detect_neighbor(i, j, cur_board)
        res = 0
        for nei in neis:
            if cur_board[nei[0]][nei[1]] == (3 - piece_type):
                res += 1
        return res

    def valid_place_check(self, previous_board, cur_board, piece_type, i, j):
        if not (i >= 0 and i < len(cur_board)):
            return False
        if not (j >= 0 and j < len(cur_board)):
            return False
        if cur_board[i][j] != 0:
            return False
        board_cur = self.copy_board(cur_board)
        board_cur[i][j] = piece_type  # play chess
        if self.find_liberty(board_cur, i, j):
            return True

        dead_pieces, board_tmp = self.remove_died_pieces(board_cur, 3 - piece_type)  # remove opponents
        if not self.find_liberty(board_tmp, i, j):
            return False
        else:
            if dead_pieces and self.compare_board_ori(previous_board, board_tmp):
                return False
        return True

    def remove_died_pieces(self, cur_board, piece_type):
        died_pieces = self.find_died_pieces(cur_board, piece_type)
        if not died_pieces: return [], cur_board
        updated_board = self.remove_certain_pieces(cur_board, died_pieces)
        return died_pieces, updated_board  # return dead_pieces and updated board

    def remove_certain_pieces(self, cur_board, positions):
        board = cur_board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        return board

    def get_candidates(self, previous_board, cur_board, piece_type):
        res = []
        for i in range(len(cur_board)):
            for j in range(len(cur_board)):
                nebs = self.detect_all_neighbours(i, j, cur_board)
                if self.valid_place_check(previous_board, cur_board, piece_type, i, j):
                    for neb in nebs:
                        if cur_board[neb[0]][neb[1]] != 0:
                            res.append((i, j))
                            break
        return res

    def detect_all_neighbours(self, i, j, cur_board):
        res = []
        diag_nei = self.survive_ways(i, j)
        ver_nei = self.detect_neighbor(i, j, cur_board)
        for candy in ver_nei:
            res.append(candy)
        for candy in diag_nei:
            res.append(candy)
        return res

    # original score with komi
    def score(self, piece_type, cur_board):
        board = cur_board
        cnt = 0
        for i in range(len(cur_board)):
            for j in range(len(cur_board)):
                if board[i][j] == piece_type:
                    cnt += 1
        if piece_type == 2:
            cnt += round(self.N / 2)
        return cnt

    def diag_two(self, mini_board, piece_type):
        top_left = mini_board[0][0]
        top_right = mini_board[0][1]
        bottom_left = mini_board[1][0]
        bottom_right = mini_board[1][1]
        condition1 = (top_left == piece_type and top_right != piece_type and bottom_left != piece_type and bottom_right == piece_type)
        condition2 = (top_left != piece_type and top_right == piece_type and bottom_left == piece_type and bottom_right != piece_type)
        return 1 if condition1 or condition2 else 0
    # evaluation of a move should consider the current advantage
    # and future influence.
    def betterScoreOne(self, piece_type, cur_board: list):
        pos_num = 0
        # pos_lib = 0
        neg_num = 0
        # neg_lib = 0
        for i in range(len(cur_board)):
            for j in range(len(cur_board)):
                if cur_board[i][j] == piece_type:
                    pos_num += 1
                    # pos_lib += self.liberty(i,j,cur_board)
                if cur_board[i][j] == 3 - piece_type:
                    neg_num += 1
                    # neg_lib += self.liberty(i,j,cur_board)
        pos_score = pos_num  # +pos_lib
        neg_score = neg_num  # +neg_num
        if piece_type == 2:
            pos_score += self.N / 2
        else:
            neg_score += self.N / 2
        return self.reward_base * (pos_score - neg_score)

    def detect_neighbor(self, i, j, cur_board):
        board = cur_board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j, cur_board):
        board = cur_board
        neighbors = self.detect_neighbor(i, j, cur_board)  # Detect neighbors
        group_allies = []
        for piece in neighbors:
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def square_one(self, mini_board: tuple, piece_type: int) -> int:
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
    def ally_dfs(self, i, j, cur_board):
        candy = [(i, j)]
        ally_members = []
        while len(candy) > 0:
            piece = candy.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1], cur_board)
            for ally in neighbor_allies:
                if ally not in candy and ally not in ally_members:
                    candy.append(ally)
        return ally_members

    # designed for score function
    def liberty(self, i, j, cur_board):
        board = cur_board
        res = 0
        ally_members = self.ally_dfs(i, j, cur_board)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1], cur_board)
            for piece in neighbors:
                if board[piece[0]][piece[1]] == 0:
                    res += 1
        return res

    # original liberty function with an input board
    def find_liberty(self, board, i, j):
        ally_members = self.ally_dfs(i, j, board)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1], board)
            for piece in neighbors:
                if board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, board, piece_type):
        died_pieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(board, i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def compare_board_ori(self, board1, board2):
        for i in range(len(board1)):
            for j in range(len(board1[i])):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self, board):
        return deepcopy(board)

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


def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"

    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1]);


if __name__ == '__main__':
    N = 5
    piece_type, pre_board, cur_board = readInput(N)
    my_player = GoPlayer(piece_type, pre_board, cur_board)
    action = my_player.minimax_pruning_with_heuristic()
    writeOutput(action)
