# player
class Player: pass

# Move (edge in the graph)
class Move: pass

# Singular game state
class GameState: pass

# Main class for which parameters will be passed
# (graph depth etc)
class Game: 
    def __init__(self, board_size, max_depth):
        self.board_size = board_size
        self.max_depth = max_depth
        
    def generate_board(self) -> GameState:
        self.game_state = GameState(self.board_size)