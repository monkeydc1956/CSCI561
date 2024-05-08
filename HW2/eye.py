def detect_eye(piece_type, cur_board, i, j):
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

def detect_secondry_eye(piece_type, cur_board, i, j):
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
    if enemy == 0 and blank <= 1:
        return True
    else:
        return False

def detect_form_eye(piece_type, cur_board, i, j):
    board = cur_board
    cur_board[i][j] = piece_type
    neighbors = []
    score = 0
    # Detect borders and add neighbor coordinates
    if i > 0: neighbors.append((i - 1, j))
    if i < len(board) - 1: neighbors.append((i + 1, j))
    if j > 0: neighbors.append((i, j - 1))
    if j < len(board) - 1: neighbors.append((i, j + 1))
    for nei in neighbors:
        if detect_eye(piece_type, cur_board, nei[0], nei[1]):
            score += 2
        elif detect_secondry_eye(piece_type, cur_board, nei[0], nei[1]):
            score += 1
    return score


# 示例用法
board = [
    [1, 2, 0, 0, 0],
    [2, 0, 2, 0, 1],
    [0, 2, 2, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0]
]
i, j = 4, 4  # 要判断的位置
player = 1  # 1表示黑棋，2表示白棋

if detect_eye(player, board, i, j):
    print("这是一个眼")
else:
    print("这不是一个眼")

if detect_secondry_eye(player, board, i, j):
    print("这是一个副眼")
else:
    print("这不是一个副眼")

print(detect_form_eye(player, board, i, j))
x = 1
y = 5
x += 2.5
print(x)
lis1 = [(1,1),(2,1)]
lis2 = [(1,2), (2,2)]
list1 = lis1 + lis2
print(list1)
#
# import math
# import random
# import time
# from read import readInput
# from copy import deepcopy
# from write import writeOutput
# import numpy as np
# from host import GO
#
#
# class myGoPlayer():
#     def __init__(self, piece_type, N):
#         self.N = N
#         self.type = "minimax_pruning"
#         self.piece_type = piece_type
#         self.reward_base = 50
#         # self.reward = [self.reward_base for i in range(N)] * N # N*N matrix, our move should always try not to move into bound position
#         self.steps = []
#         self.max_depth = 1
#         self.reward = []
#         self.max_step = 23
#         self.X_CHANGES = [1, 0, -1, 0]
#         self.Y_CHANGES = [0, 1, 0, -1]
#         # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
#         # self.reward.append((-self.reward_base, 0, self.reward_base, 0, -self.reward_base))
#         # self.reward.append((0, self.reward_base, self.reward * 2, self.reward_base, 0))
#         # self.reward.append((-self.reward_base, 0, self.reward_base, 0, -self.reward_base))
#         # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
#         # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
#         # self.reward.append((-self.reward_base, 5, self.reward_base, 5, -self.reward_base))
#         # self.reward.append((0, self.reward_base, self.reward * 2, self.reward_base, 0))
#         # self.reward.append((-self.reward_base, 5, self.reward_base, 5, -self.reward_base))
#         # self.reward.append((-2 * self.reward_base, -self.reward_base, 0, -self.reward_base, -2 * self.reward_base))
#         self.reward.append((-100, 0, 5, 0, -100))
#         self.reward.append((0, 5, 10, 5, 0))
#         self.reward.append((5, 10, 100, 10, 5))
#         self.reward.append((0, 5, 10, 5, 0))
#         self.reward.append((-100, 0, 5, 0, -100))
#         # -100 -50 0 -50 -100
#         # -50 0 50 0 -50
#         # 0 50 100 50 0
#         # -50 0 50 0 -50
#         # -100 -50 0 -50 -100
#         # 1 = white | 2 = black
#
#     # pseudocode
#     # def minimax_alpha_beta(board, depth, maximizing_player, alpha, beta):
#     #     if depth == 0 or game_over(board):
#     #     if maximizing_player:
#     #         max_eval = -float('inf')
#     #         for move in get_possible_moves(board):
#     #             eval = minimax_alpha_beta(board_after_move, depth - 1, False, alpha, beta)
#     #             max_eval = max(max_eval, eval)
#     #             alpha = max(alpha, eval)
#     #             if beta <= alpha:
#     #                 break  # Alpha-Beta pruning
#     #         return max_eval
#     #     else:
#     #         min_eval = float('inf')
#     #         for move in get_possible_moves(board):
#     #             eval = minimax_alpha_beta(board_after_move, depth - 1, True, alpha, beta)
#     #             min_eval = min(min_eval, eval)
#     #             beta = min(beta, eval)
#     #             if beta <= alpha:
#     #                 break  # Alpha-Beta pruning
#     #         return min_eval
#
#     def minimax(self, piece_tupe, previous_board, cur_board, cur_steps):
#         # root always choose max value, it is an actual max layer
#         # board_cur = deepcopy(cur_board)
#         board_previous = self.copy_board(cur_board)  # now become the past
#         next_step_set = self.get_candidates(previous_board, board_previous, piece_type)
#         res = -math.inf
#         # pre_enemies = self.score(3 - piece_type, cur_board)
#         best_move = "PASS"
#         if not next_step_set:
#             return
#         if len(next_step_set) <= 3:
#             self.max_depth = 3
#         for step in next_step_set:  # first enter is max move therefore we start with min
#             # eye = 0
#             # secondry_eye = 0
#             # enemy_secondry_eye = 0
#             # do not block own eyes
#             # if self.detect_eye(piece_type,cur_board,step[0],step[1]):
#             #     eye = 1
#             # if self.detect_secondry_eye(piece_type,cur_board,step[0],step[1]):
#             #     secondry_eye = 1
#             # # try to steal enemy eye
#             # if self.detect_secondry_eye(3-piece_type, cur_board, step[0], step[1]):
#             #     enemy_secondry_eye = 1
#
#             board_cur = self.copy_board(cur_board)  # now become the future
#             board_cur[step[0]][step[1]] = piece_type
#             _, board_cur = self.remove_died_pieces(board_cur, 3 - piece_type)
#             next_score = self.minVal(self.max_depth, board_previous, board_cur, 3 - piece_tupe, -math.inf, math.inf,
#                                      cur_steps)  # second layer lack pruning in this way
#             # kills = self.around_enemies(step[0], step[1], piece_type, cur_board)
#
#             ## other heuristic
#             # killed_enemies = pre_enemies - self.score(3 - piece_type, board_cur)
#             #
#             # liberty_score = self.liberty(step[0], step[1], board_cur)#
#             # if liberty_score >= 2:
#             #     next_score += liberty_score*self.reward_base
#             # else:
#             #     next_score -= liberty_score*self.reward_base
#
#             # final_score = next_score + self.future_adv(step[0], step[1], board_cur) + self.reward[step[0]][
#             #     step[1]] * 1 + 1000 * killed_enemies + kills * 10 + -800 * eye + -150*secondry_eye + 150*enemy_secondry_eye
#             ## other heuristic
#             # final_score = next_score + self.future_adv(step[0], step[1], board_cur) + self.reward[step[0]][
#             #     step[1]] * 0.1 + kills
#             final_score = next_score
#             if final_score > res:
#                 res = final_score
#                 best_move = step
#         return best_move
#
#     def minVal(self, depth, previous_board, cur_board, piece_tupe, alpha, beta, steps):
#         board_previous = self.copy_board(cur_board)  # now become the past
#         next_step_set = self.get_candidates(previous_board, board_previous, piece_type)
#         if depth == 0 or not next_step_set or steps > self.max_step:
#             return self.evaluate_game_state(cur_board, piece_tupe)
#             # return self.betterScoreOne(piece_tupe, cur_board)
#         res = math.inf
#         for step in next_step_set:
#             board_cur = self.copy_board(cur_board)
#             board_cur[step[0]][step[1]] = piece_type
#             _, board_cur = self.remove_died_pieces(board_cur, 3 - piece_type)
#             next_score = self.maxVal(depth - 1, board_previous, board_cur, 3 - piece_tupe, alpha, beta, steps + 1)
#             if next_score < res:
#                 res = next_score
#             if res <= alpha:
#                 return res
#             if res < beta:
#                 beta = res
#         return res
#
#     def maxVal(self, depth, previous_board, cur_board, piece_tupe, alpha, beta, steps):
#         board_previous = self.copy_board(cur_board)  # now become the past
#         next_step_set = self.get_candidates(previous_board, board_previous, piece_type)
#         if depth == 0 or not next_step_set or steps > self.max_step:
#             # return self.betterScoreOne(piece_tupe, cur_board)
#             return self.evaluate_game_state(cur_board, piece_tupe)
#         res = -math.inf
#         for step in next_step_set:
#             board_cur = self.copy_board(cur_board)
#             board_cur[step[0]][step[1]] = piece_type
#             _, board_cur = self.remove_died_pieces(board_cur, 3 - piece_type)
#             next_score = self.minVal(depth - 1, board_previous, board_cur, 3 - piece_tupe, alpha, beta, steps + 1)
#             if res < next_score:
#                 res = next_score
#             if res >= beta:
#                 return res
#             if res > alpha:
#                 alpha = res
#         return res
#
#     def move(self, previous_board, cur_board):
#         stones = 0
#         for i in range(len(cur_board)):
#             for j in range(len(cur_board)):
#                 if cur_board[i][j] != 0:
#                     stones += 1
#         if (stones == 0 or stones == 1) and cur_board[2][2] == 0:
#             return (2, 2)  # always trys to get central pos
#         # if stones == 1 and cur_board[2][2] == 1:
#         #     return (2, 1)  # "stick" strategy and save time
#
#         return self.minimax(self.piece_type, previous_board, cur_board, stones)
#
#     def detect_eye(self, piece_type, cur_board, i, j):
#         board = cur_board
#         neighbors = []
#         # Detect borders and add neighbor coordinates
#         if i > 0: neighbors.append((i - 1, j))
#         if i < len(board) - 1: neighbors.append((i + 1, j))
#         if j > 0: neighbors.append((i, j - 1))
#         if j < len(board) - 1: neighbors.append((i, j + 1))
#         enemy = 0
#         blank = 0
#         for nei in neighbors:
#             if cur_board[nei[0]][nei[1]] == 3 - piece_type:
#                 enemy += 1
#             if cur_board[nei[0]][nei[1]] == 0:
#                 blank += 1
#         if enemy == 0 and blank == 0:
#             return True
#         else:
#             return False
#
#     def detect_secondry_eye(self, piece_type, cur_board, i, j):
#         board = cur_board
#         neighbors = []
#         # Detect borders and add neighbor coordinates
#         if i > 0: neighbors.append((i - 1, j))
#         if i < len(board) - 1: neighbors.append((i + 1, j))
#         if j > 0: neighbors.append((i, j - 1))
#         if j < len(board) - 1: neighbors.append((i, j + 1))
#         enemy = 0
#         blank = 0
#         for nei in neighbors:
#             if cur_board[nei[0]][nei[1]] == 3 - piece_type:
#                 enemy += 1
#             if cur_board[nei[0]][nei[1]] == 0:
#                 blank += 1
#         if enemy == 0 and blank <= 2:
#             return True
#         else:
#             return False
#
#     def future_adv(self, i, j, cur_board):
#         return self.sur_score(i, j, cur_board) + self.ally_score(i, j, cur_board)
#
#     def sur_score(self, i, j, cur_board):
#         score = 0
#         for nei in self.detect_neighbor(i, j, cur_board):
#             if cur_board[nei[0]][nei[1]] == 0:
#                 score += 1
#         return score
#
#     def eye_score(self, i, j, cur_board):
#         score = 0
#         return score
#
#     def ally_score(self, i, j, cur_board):
#         score = 0
#         for friend in self.survive_ways(i, j):
#             if cur_board[friend[0]][friend[1]] == self.piece_type:
#                 score += 1
#         return score
#
#     def survive_ways(self, i, j):
#         diags = []
#         if i - 1 >= 0 and j - 1 >= 0:
#             diags.append((i - 1, j - 1))
#         if i - 1 >= 0 and j + 1 < self.N:
#             diags.append((i - 1, j + 1))
#         if i + 1 < self.N and j - 1 >= 0:
#             diags.append((i + 1, j - 1))
#         if i + 1 < self.N and j + 1 < self.N:
#             diags.append((i + 1, j + 1))
#         return diags
#
#     def around_enemies(self, i, j, piece_type, cur_board):
#         neis = self.detect_neighbor(i, j, cur_board)
#         res = 0
#         for nei in neis:
#             if cur_board[nei[0]][nei[1]] == (3 - piece_type):
#                 res += 1
#         return res
#
#     def valid_place_check(self, previous_board, cur_board, piece_type, i, j):
#         if not (i >= 0 and i < len(cur_board)):
#             return False
#         if not (j >= 0 and j < len(cur_board)):
#             return False
#         if cur_board[i][j] != 0:
#             return False
#         board_cur = self.copy_board(cur_board)
#         board_cur[i][j] = piece_type  # play chess
#         if self.find_liberty(board_cur, i, j):
#             return True
#
#         dead_pieces, board_tmp = self.remove_died_pieces(board_cur, 3 - piece_type)  # remove opponents
#         if not self.find_liberty(board_tmp, i, j):
#             return False
#         else:
#             if dead_pieces and self.compare_board(previous_board, board_tmp):
#                 return False
#         return True
#
#     def remove_died_pieces(self, cur_board, piece_type):
#         died_pieces = self.find_died_pieces(cur_board, piece_type)
#         if not died_pieces: return [], cur_board
#         updated_board = self.remove_certain_pieces(cur_board, died_pieces)
#         return died_pieces, updated_board  # return dead_pieces and updated board
#
#     def remove_certain_pieces(self, cur_board, positions):
#         board = cur_board
#         for piece in positions:
#             board[piece[0]][piece[1]] = 0
#         return board
#
#     def get_candidates(self, previous_board, cur_board, piece_type):
#         res = []
#         for i in range(len(cur_board)):
#             for j in range(len(cur_board)):
#                 nebs = self.detect_all_neighbours(i, j, cur_board)
#                 if self.valid_place_check(previous_board, cur_board, piece_type, i, j):
#                     for neb in nebs:
#                         if cur_board[neb[0]][neb[1]] != 0:
#                             res.append((i, j))
#                             break
#         return res
#
#     def detect_all_neighbours(self, i, j, cur_board):
#         res = []
#         diag_nei = self.survive_ways(i, j)
#         ver_nei = self.detect_neighbor(i, j, cur_board)
#         for candy in ver_nei:
#             res.append(candy)
#         for candy in diag_nei:
#             res.append(candy)
#         return res
#
#     # original score with komi
#     def score(self, piece_type, cur_board):
#         board = cur_board
#         cnt = 0
#         for i in range(len(cur_board)):
#             for j in range(len(cur_board)):
#                 if board[i][j] == piece_type:
#                     cnt += 1
#         if piece_type == 2:
#             cnt += round(self.N / 2)
#         return cnt
#
#     # evaluation of a move should consider the current advantage
#     # and future influence.
#     def betterScoreOne(self, piece_type, cur_board: list):
#         pos_num = 0
#         # pos_lib = 0
#         neg_num = 0
#         # neg_lib = 0
#         for i in range(len(cur_board)):
#             for j in range(len(cur_board)):
#                 if cur_board[i][j] == piece_type:
#                     pos_num += 1
#                     # pos_lib += self.liberty(i,j,cur_board)
#                 if cur_board[i][j] == 3 - piece_type:
#                     neg_num += 1
#                     # neg_lib += self.liberty(i,j,cur_board)
#         pos_score = pos_num  # +pos_lib
#         neg_score = neg_num  # +neg_num
#         if piece_type == 2:
#             pos_score += self.N / 2
#         else:
#             neg_score += self.N / 2
#         return self.reward_base * (pos_score - neg_score)
#         # return pos_num+pos_lib-neg_num-neg_lib
#
#     # evaluation of a move should consider the current advantage
#     # and how many enemies it could kill.
#
#     def detect_neighbor(self, i, j, cur_board):
#         board = cur_board
#         neighbors = []
#         # Detect borders and add neighbor coordinates
#         if i > 0: neighbors.append((i - 1, j))
#         if i < len(board) - 1: neighbors.append((i + 1, j))
#         if j > 0: neighbors.append((i, j - 1))
#         if j < len(board) - 1: neighbors.append((i, j + 1))
#         return neighbors
#
#     def detect_neighbor_ally(self, i, j, cur_board):
#         board = cur_board
#         neighbors = self.detect_neighbor(i, j, cur_board)  # Detect neighbors
#         group_allies = []
#         # Iterate through neighbors
#         for piece in neighbors:
#             # Add to allies list if having the same color
#             if board[piece[0]][piece[1]] == board[i][j]:
#                 group_allies.append(piece)
#         return group_allies
#
#     def ally_dfs(self, i, j, cur_board):
#         stack = [(i, j)]  # stack for DFS serach
#         ally_members = []  # record allies positions during the search
#         while stack:
#             piece = stack.pop()
#             ally_members.append(piece)
#             neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1], cur_board)
#             for ally in neighbor_allies:
#                 if ally not in stack and ally not in ally_members:
#                     stack.append(ally)
#         return ally_members
#
#     # designed for score function
#     def liberty(self, i, j, cur_board):
#         board = cur_board
#         res = 0
#         ally_members = self.ally_dfs(i, j, cur_board)
#         for member in ally_members:
#             neighbors = self.detect_neighbor(member[0], member[1], cur_board)
#             for piece in neighbors:
#                 # If there is empty space around a piece, it has liberty
#                 if board[piece[0]][piece[1]] == 0:
#                     res += 1
#         # If none of the pieces in a allied group has an empty space, it has no liberty
#         return res
#
#     # original liberty function with an input board
#     def find_liberty(self, board, i, j):
#         ally_members = self.ally_dfs(i, j, board)
#         for member in ally_members:
#             neighbors = self.detect_neighbor(member[0], member[1], board)
#             for piece in neighbors:
#                 # If there is empty space around a piece, it has liberty
#                 if board[piece[0]][piece[1]] == 0:
#                     return True
#         # If none of the pieces in a allied group has an empty space, it has no liberty
#         return False
#
#     # Find the died pieces that has no liberty in the board for a given piece type (from host.py, slightly modified)
#     def find_died_pieces(self, board, piece_type):
#         died_pieces = []
#         for i in range(len(board)):
#             for j in range(len(board)):
#                 # Check if there is a piece at this position:
#                 if board[i][j] == piece_type:
#                     # The piece die if it has no liberty
#                     if not self.find_liberty(board, i, j):
#                         died_pieces.append((i, j))
#         return died_pieces
#
#     def compare_board(self, board1, board2):
#         for i in range(len(board1)):
#             for j in range(len(board1[i])):
#                 if board1[i][j] != board2[i][j]:
#                     return False
#         return True
#
#     # Copy the current board for potential testing (from host.py, slightly modified)
#     def copy_board(self, board):
#         return deepcopy(board)
#
#     def evaluate_game_state(self, game_state, side):
#         # Define heuristic here
#         # Count number of sides stones - opponent stones
#         opponent_side = 3 - side
#         side_count = 0
#         side_liberty = set()
#         opponent_count = 0
#         opponent_liberty = set()
#         for i in range(self.N):
#             for j in range(self.N):
#                 if game_state[i][j] == side:
#                     side_count += 1
#                 elif game_state[i][j] == opponent_side:
#                     opponent_count += 1
#                 # This point should be UNOCCUPIED!
#                 else:
#                     for index in range(len(self.X_CHANGES)):
#                         new_i = i + self.X_CHANGES[index]
#                         new_j = j + self.Y_CHANGES[index]
#                         if 0 <= new_i < self.N and 0 <= new_j < self.N:
#                             if game_state[new_i][new_j] == side:
#                                 side_liberty.add((i, j))
#                             elif game_state[new_i][new_j] == opponent_side:
#                                 opponent_liberty.add((i, j))
#
#         side_edge_count = 0
#         opponent_side_edge_count = 0
#         for j in range(self.N):
#             if game_state[0][j] == side or game_state[self.N - 1][j] == side:
#                 side_edge_count += 1
#             if game_state[0][j] == opponent_side or game_state[self.N - 1][j] == opponent_side:
#                 opponent_side_edge_count += 1
#
#         for j in range(1, self.N - 1):
#             if game_state[j][0] == side or game_state[j][self.N - 1] == side:
#                 side_edge_count += 1
#             if game_state[j][0] == opponent_side or game_state[j][self.N - 1] == opponent_side:
#                 opponent_side_edge_count += 1
#
#         center_unoccupied_count = 0
#         for i in range(1, self.N - 1):
#             for j in range(1, self.N - 1):
#                 if game_state[i][j] == 0:
#                     center_unoccupied_count += 1
#
#         score = min(max((len(side_liberty) - len(opponent_liberty)), -8), 8) + (
#                 -4 * self.calculate_euler_number(game_state, side)) + (
#                         5 * (side_count - opponent_count)) - (9 * side_edge_count * (center_unoccupied_count / 9))
#         if self.piece_type == 1:
#             score += 2.5
#         return score
#
#     def calculate_euler_number(self, game_state, side):
#         opponent_side = 3 - side
#         new_game_state = np.zeros((self.N + 2, self.N + 2), dtype=int)
#         # First copy the original game_state
#         for i in range(self.N):
#             for j in range(self.N):
#                 new_game_state[i + 1][j + 1] = game_state[i][j]
#
#         q1_side = 0
#         q2_side = 0
#         q3_side = 0
#         q1_opponent_side = 0
#         q2_opponent_side = 0
#         q3_opponent_side = 0
#
#         for i in range(self.N):
#             for j in range(self.N):
#                 new_game_sub_state = new_game_state[i: i + 2, j: j + 2]
#                 q1_side += self.count_q1(new_game_sub_state, side)
#                 q2_side += self.count_q2(new_game_sub_state, side)
#                 q3_side += self.count_q3(new_game_sub_state, side)
#                 q1_opponent_side += self.count_q1(new_game_sub_state, opponent_side)
#                 q2_opponent_side += self.count_q2(new_game_sub_state, opponent_side)
#                 q3_opponent_side += self.count_q3(new_game_sub_state, opponent_side)
#
#         return (q1_side - q3_side + 2 * q2_side - (q1_opponent_side - q3_opponent_side + 2 * q2_opponent_side)) / 4
#
#     def count_q1(self, game_sub_state, side):
#         if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
#              and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
#                     and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
#                     and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
#                     and game_sub_state[1][0] != side and game_sub_state[1][1] == side)):
#             return 1
#         else:
#             return 0
#
#     def count_q2(self, game_sub_state, side):
#         if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
#              and game_sub_state[1][0] != side and game_sub_state[1][1] == side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
#                     and game_sub_state[1][0] == side and game_sub_state[1][1] != side)):
#             return 1
#         else:
#             return 0
#
#     def count_q3(self, game_sub_state, side):
#         if ((game_sub_state[0][0] == side and game_sub_state[0][1] == side
#              and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
#                     and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
#                 or (game_sub_state[0][0] == side and game_sub_state[0][1] != side
#                     and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
#                 or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
#                     and game_sub_state[1][0] == side and game_sub_state[1][1] == side)):
#             return 1
#         else:
#             return 0
#
#
# if __name__ == "__main__":
#     N = 5
#     piece_type, pre_board, cur_board = readInput(N)
#     go = GO(N)
#     go.set_board(piece_type, pre_board, cur_board)
#     begin_time = time.time()
#     myPlayer = myGoPlayer(piece_type, N)
#     move = myPlayer.move(pre_board, cur_board)
#     writeOutput(move)
#     print(time.time() - begin_time)