from copy import deepcopy
class GO:
    def __init__(self, n):
        self.cur_piece_type = 0
        self.size = n
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.verbose = False

    def nextGoBoard(self, i, j, piece_type):
        go_future = GO(self.size)
        previous_board_future = self.board # new is previous
        board_future = deepcopy(previous_board_future) # future is now
        board_future[i][j] = piece_type
        go_future.set_board(3 - piece_type, previous_board_future, board_future)
        go_future.remove_died_pieces(3 - piece_type)
        return go_future

    def candidates(self):
        possible_placements = []
        for i in range(self.size):
            for j in range(self.size):
                surrondings = self.detect_all_neighbours(i, j)
                if self.valid_place_check(self.cur_piece_type, i, j):
                    for stone in surrondings:
                        if self.board[stone[0]][stone[1]] != 0:
                            possible_placements.append((i, j))
                            break

        return possible_placements

    # get score of one player based on its number
    def betterScoreOne(self, piece_type):
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    score += 1
        if piece_type == 2:
            score += self.size / 2
        return score

    # evaluation score of current board
    def heuristic(self):
        return 50 * (self.betterScoreOne(self.cur_piece_type) - self.betterScoreOne(3 - self.cur_piece_type))

    # liberty factor of given location
    def get_liberty_score(self, i, j):
        score = 0
        factor = 1
        for nei in self.detect_neighbor(i, j):
            if self.board[nei[0]][nei[1]] == 0:
                score = score + factor
        return score

    # ally factor of given location
    def get_ally_score(self, i, j):
        score = 0
        diag_nei_factor = 1
        for nei in self.detect_diag_nieighbours(i, j):
            if self.board[nei[0]][nei[1]] == self.cur_piece_type:
                score = score + diag_nei_factor
        return score

    # location evaluation score
    def adv_future(self, i, j):
        return self.get_liberty_score(i, j) + self.get_ally_score(i, j)

    # detect diag neis
    def detect_diag_nieighbours(self, i, j):
        neighbours = []
        if i - 1 >= 0 and j - 1 >= 0:
            neighbours.append((i - 1, j - 1))
        if i - 1 >= 0 and j + 1 < self.size:
            neighbours.append((i - 1, j + 1))
        if i + 1 < self.size and j - 1 >= 0:
            neighbours.append((i + 1, j - 1))
        if i + 1 < self.size and j + 1 < self.size:
            neighbours.append((i + 1, j + 1))
        return neighbours

    # detect all neis of give location
    def detect_all_neighbours(self, i, j):
        neighbours = []
        diag_nei = self.detect_diag_nieighbours(i, j)
        other_nei = self.detect_neighbor(i, j)
        for nei in other_nei:
            neighbours.append(nei)
        for nei in diag_nei:
            neighbours.append(nei)
        return neighbours

    # detect number of neighbour enemies
    def survive_ways(self,i, j, piece_type):
        neis = self.detect_neighbor(i, j)
        re = 0
        for nei in neis:
            if self.board[nei[0]][nei[1]] == (3 - piece_type):
                re = re + 1
        return re

    def init_board(self, n):
        board = [[0 for x in range(n)] for y in range(n)]
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        self.cur_piece_type = piece_type
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def valid_place_check(self, piece_type, i, j):
        cur_board = self.board
        if not (i >= 0 and i < len(cur_board)):
            return False
        if not (j >= 0 and j < len(cur_board)):
            return False
        if cur_board[i][j] != 0:
            return False
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                return False
        return True

    def update_board(self, new_board):
        self.board = new_board

    def score(self, piece_type):
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def candidates(self):
        possible_placements = []
        for i in range(self.size):
            for j in range(self.size):
                neis = self.detect_all_neighbours(i, j)
                if self.valid_place_check(self.cur_piece_type, i, j):
                    for nei in neis:
                        if self.board[nei[0]][nei[1]] != 0:
                            possible_placements.append((i, j))
                            break

        return possible_placements

    # get score of one player based on its number
    def get_score(self, piece_type):
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    score += 1
        if piece_type == 2:
            score += self.size / 2
        return score

    # evaluation score of current board
    def heuristic(self):
        return 50 * (self.get_score(self.cur_piece_type) - self.get_score(3 - self.cur_piece_type))

    # liberty factor of given location
    def get_liberty_score(self, i, j):
        score = 0
        factor = 1
        for nei in self.detect_neighbor(i, j):
            if self.board[nei[0]][nei[1]] == 0:
                score = score + factor
        return score

    # ally factor of given location
    def get_ally_score(self, i, j):
        score = 0
        # nei_factor = 6
        diag_nei_factor = 1
        # for nei in self.detect_neighbor(i, j):
        #     if self.board[nei[0]][nei[1]] == self.piece_type:
        #         score = score + nei_factor
        for nei in self.detect_diag_nieighbours(i, j):
            if self.board[nei[0]][nei[1]] == self.cur_piece_type:
                score = score + diag_nei_factor
        return score

    # location evaluation score
    def loc_evaluation(self, i, j):
        return self.get_liberty_score(i, j) + self.get_ally_score(i, j)

    # detect diag neis
    def detect_diag_nieighbours(self, i, j):
        neighbours = []
        if i - 1 >= 0 and j - 1 >= 0:
            neighbours.append((i - 1, j - 1))
        if i - 1 >= 0 and j + 1 < self.size:
            neighbours.append((i - 1, j + 1))
        if i + 1 < self.size and j - 1 >= 0:
            neighbours.append((i + 1, j - 1))
        if i + 1 < self.size and j + 1 < self.size:
            neighbours.append((i + 1, j + 1))
        return neighbours

    # detect all neis of give location
    def detect_all_neighbours(self, i, j):
        neighbours = []
        diag_nei = self.detect_diag_nieighbours(i, j)
        other_nei = self.detect_neighbor(i, j)
        for nei in other_nei:
            neighbours.append(nei)
        for nei in diag_nei:
            neighbours.append(nei)
        return neighbours

    # detect number of neighbour enemies
    def detect_num_nei_enemies(self,i, j, piece_type):
        neis = self.detect_neighbor(i, j)
        re = 0
        for nei in neis:
            if self.board[nei[0]][nei[1]] == (3 - piece_type):
                re = re + 1
        return re