import copy
from enum import Enum
from typing import List, Tuple, Optional, Callable

# Game State and Move Generation

def generate_board(m, n):
    board = []
    current_white = False
    for i in range(m):
        board.append([])
        for j in range(n):
            if current_white: board[i].append('W')
            else: board[i].append('B')
            current_white = not current_white
        current_white = not current_white
    return board

class Player(Enum):
    BLACK = 'B'
    WHITE = 'W'

    @property
    def opponent(self):
        return Player.WHITE if self == Player.BLACK else Player.BLACK

Position = Tuple[int, int]
Move = Tuple[Position, Position]  # from_pos, to_pos

class GameState:
    def __init__(self, board: List[List[str]], current: Player):
        self.board = board
        self.current = current
        self.rows = len(board)
        self.cols = len(board[0]) if board else 0

    def get_valid_moves(self) -> List[Move]:
        """Generate all clobber moves for the current player."""
        moves: List[Move] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self.current.value:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self.board[nr][nc] == self.current.opponent.value:
                                moves.append(((r, c), (nr, nc)))
        return moves

    def make_move(self, move: Move) -> 'GameState':
        """Return a new GameState after applying the given move."""
        (r1, c1), (r2, c2) = move
        new_board = copy.deepcopy(self.board)
        new_board[r2][c2] = new_board[r1][c1]
        new_board[r1][c1] = '_'
        return GameState(new_board, self.current.opponent)

    def is_terminal(self) -> bool:
        """Game ends when no moves are available."""
        return len(self.get_valid_moves()) == 0

# Heuristics

def heuristic_mobility(state: GameState, player: Player) -> int:
    """Heuristic H1: Number of available moves for 'player'."""
    original = state.current
    state.current = player
    count = len(state.get_valid_moves())
    state.current = original
    return count

def heuristic_diff_mobility(state: GameState, player: Player) -> int:
    """Heuristic H2: Difference between player's and opponent's moves."""
    return heuristic_mobility(state, player) - heuristic_mobility(state, player.opponent)

def heuristic_piece_count(state: GameState, player: Player) -> int:
    """Heuristic H3: Difference in number of stones."""
    cnt_self = sum(row.count(player.value) for row in state.board)
    cnt_opp = sum(row.count(player.opponent.value) for row in state.board)
    return cnt_self - cnt_opp

# Minimax Implementation

def minimax(state: GameState,
            depth: int,
            heuristic: Callable[[GameState, Player], int],
            maximizing: bool = True
           ) -> Tuple[int, Optional[Move]]:
    if depth == 0 or state.is_terminal():
        return heuristic(state, Player.BLACK), None

    moves = state.get_valid_moves()
    best_move = None

    if maximizing:
        max_eval = float('-inf')
        for mv in moves:
            val, _ = minimax(state.make_move(mv), depth-1, heuristic, False)
            if val > max_eval:
                max_eval, best_move = val, mv
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for mv in moves:
            val, _ = minimax(state.make_move(mv), depth-1, heuristic, True)
            if val < min_eval:
                min_eval, best_move = val, mv
        return min_eval, best_move

# Alpha-Beta Pruning Implementation

def alphabeta(state: GameState,
              depth: int,
              alpha: float,
              beta: float,
              heuristic: Callable[[GameState, Player], int],
              maximizing: bool = True
             ) -> Tuple[int, Optional[Move]]:
    if depth == 0 or state.is_terminal():
        return heuristic(state, Player.BLACK), None

    moves = state.get_valid_moves()
    best_move = None

    if maximizing:
        value = float('-inf')
        for mv in moves:
            val, _ = alphabeta(state.make_move(mv), depth-1, alpha, beta, heuristic, False)
            if val > value:
                value, best_move = val, mv
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = float('inf')
        for mv in moves:
            val, _ = alphabeta(state.make_move(mv), depth-1, alpha, beta, heuristic, True)
            if val < value:
                value, best_move = val, mv
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move

