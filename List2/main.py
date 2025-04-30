import sys
import time
from clobber_game import *

def parse_input():
    m, n = map(int, input("Board size (rows cols): ").split(' '))
    generate_board_answer = input("Do you want to generate the board? (Yes)")
    if generate_board_answer.lower()=="yes" or not generate_board_answer:
        board = generate_board(m, n)
    else:
        print("Now paste the board:")
        board = [input().split() for _ in range(m)]
    
    if len(board)!=m: raise ValueError("Wrong amount of rows!")
    for i in range(m):
        if len(board[i])!=n: raise ValueError("Wrong amount of elements in row!")
    
    print("Heuristics: 1) Mobility  2) DiffMob  3) PieceCountDiff", file=sys.stderr)
    h = int(input())
    print("Algorithm: 1) Minimax  2) AlphaBeta", file=sys.stderr)
    a = int(input())
    print("Choose depth: ", end='')
    d = int(input())
    return board, h, a, d

def select_heuristic(i):
    return {
        1: heuristic_mobility,
        2: heuristic_diff_mobility,
        3: heuristic_piece_count
    }[i]

def select_algorithm(i):
    if i == 1:
        return minimax
    else:
        # wrap alphabeta so signature matches: (state, depth, heuristic, maximizing)
        return lambda state, depth, h, maxp: alphabeta(
            state, depth, float('-inf'), float('inf'), h, maxp
        )

def print_board(board):
    for row in board:
        print(" ".join(row))

def main():
    board, h_choice, a_choice, depth = parse_input()
    heuristic = select_heuristic(h_choice)
    algorithm = select_algorithm(a_choice)

    start_time = time.perf_counter()

    state = GameState(board, Player.BLACK)
    maximizing = True
    rounds = 0

    while not state.is_terminal():
        _, move = algorithm(state, depth, heuristic, maximizing)
        if move is None:
            break
        state = state.make_move(move)
        maximizing = not maximizing
        rounds += 1

    end_time = time.perf_counter()

    # stdout: final board + winner/rounds
    print_board(state.board)
    winner = state.current.opponent.value
    print(f"Winner: {winner} Rounds: {rounds}")

    # stderr: nodes and time
    total_time = end_time - start_time
    print(f"Running time: {total_time:.6f} s")

if __name__ == "__main__":
    main()
