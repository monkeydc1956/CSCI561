import random
import time
import math
from read import readInput
from write import writeOutput
import os

from host import GO
from newGo import GO as newGoBoard


class myPlayer():
    def __init__(self, go, piece_type):
        self.type = 'random'
        self.go = go
        self.piece_type = piece_type
        self.int_max = math.inf
        self.int_min = -math.inf
        self.reward = []
        self.reward.append((-100, 0, 5, 0, -100))
        self.reward.append((0, 5, 10, 5, 0))
        self.reward.append((5, 10, 100, 10, 5))
        self.reward.append((0, 5, 10, 5, 0))
        self.reward.append((-100, 0, 5, 0, -100))
        self.movements = []
        self.max_depth = 1
        self.max_steps = 23 # escape dump step at end

    # return minimax solution
    def minimax(self, step):
        best_score = self.int_min
        pre_enemies = self.score(3 - self.piece_type, self.go.board)
        candidates = go.candidates()
        best_move = "PASS"
        for move in candidates:
            flag_eye = 0
            flag_secondary_eye = 0
            if self.detect_eye(self.piece_type, self.go.board, move[0], move[1]):
                flag_eye = 1
            if self.detect_eye(self.piece_type, self.go.board, move[0], move[1]):
                flag_secondary_eye = 1
            enemy_secondry_eye = 0
            if self.detect_secondry_eye(3-self.piece_type, self.go.board, move[0], move[1]):
                enemy_secondry_eye = 1

            n_go = newGoBoard.NextGoBoard(go, move[0], move[1], self.piece_type)
            enemies_future = self.score(3 - self.piece_type, n_go.board)
            n_score = self.minVal(n_go, 1, 3 - self.piece_type, self.int_min, self.int_max, step)
            kills = go.detect_num_nei_enemies(move[0], move[1], piece_type)
            score = (n_score + n_go.loc_evaluation(move[0], move[1]) + self.reward[move[0]][move[1]] + kills*100 +
                     (-1000*flag_eye) + (-100*flag_secondary_eye) + 100*enemy_secondry_eye) + 50*(pre_enemies - enemies_future) + 50*self.detect_form_eye(self.piece_type, self.go.board, move[0], move[1])
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
        # return random.choice(possible_placements)

    def minVal(self, go, depth, piece_type, alpha, beta, step):
        # when depth is over max_depth or step over max_step, return current heuristic val
        if depth > self.max_depth or step > self.max_steps:
            return go.heuristic()
        candidates = go.candidates()
        if not candidates:
            return go.heuristic()
        # check all moves
        pre_enemies = self.score(3 - self.piece_type, self.go.board)
        best_value = self.int_max
        for move in candidates:
            n_go = newGoBoard.NextGoBoard(go, move[0], move[1], piece_type)
            enemies_future = self.score(3 - self.piece_type, n_go.board)
            n_score = self.maxVal(n_go, depth + 1, 3 - piece_type, alpha, beta, step+1) + 50*(pre_enemies - enemies_future)
            if n_score < best_value:
                best_value = n_score
            if best_value <= alpha:
                return best_value
            if best_value < beta:
                beta = best_value
        return best_value

    def maxVal(self, go, depth, piece_type, alpha, beta, step):
        # when depth is over max_depth or step over max_step, return current heuristic val
        if depth > self.max_depth or step > self.max_steps:
            return go.heuristic()
        # get all possible placements, if there is none, return current heuristic val
        candidates = go.candidates()
        if not candidates:
            return go.heuristic()
        # check all moves
        pre_enemies = self.score(3 - self.piece_type, self.go.board)
        best_value = self.int_min
        for move in candidates:
            n_go = newGoBoard.NextGoBoard(go, move[0], move[1], piece_type)
            enemies_future = self.score(3 - self.piece_type, n_go.board)
            n_score = self.minVal(n_go, depth + 1, 3 - piece_type, alpha, beta, step+1) + 50*(pre_enemies - enemies_future)
            if n_score > best_value:
                best_value = n_score
            if best_value >= beta:
                return best_value
            if best_value > alpha:
                alpha = best_value
        return best_value

    def score(self, piece_type, cur_board):
        board = cur_board
        cnt = 0
        for i in range(len(cur_board)):
            for j in range(len(cur_board)):
                if board[i][j] == piece_type:
                    cnt += 1
        if piece_type == 2:
            cnt += len(cur_board) / 2
        return cnt

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
        if enemy == 0 and blank <= 1:
            return True
        else:
            return False

    def detect_form_eye(self, piece_type, cur_board, i, j):
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
            if self.detect_eye(piece_type, cur_board, nei[0], nei[1]):
                score += 2
            elif self.detect_secondry_eye(piece_type, cur_board, nei[0], nei[1]):
                score += 1
        return score
    def move(self):
        stones = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] != 0:
                    stones += 1
        if (stones == 0 or stones == 1) and go.board[2][2] == 0: # always take central point
            return (2, 2)
        return self.minimax(stones)



if __name__ == "__main__":
    start = time.time()
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = newGoBoard(N)
    go.set_board(piece_type, previous_board, board)
    player = myPlayer(go, piece_type)
    action = player.move()
    writeOutput(action)