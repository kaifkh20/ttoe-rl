import random, pickle

# -------- Tic-Tac-Toe Setup --------
def initialize_board():
    return [" "] * 9

def check_winner(board):
    win_patterns = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for pattern in win_patterns:
        if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != " ":
            return board[pattern[0]]
    return None

def is_full(board): return " " not in board
def get_state(board): return "".join(board)
def available_moves(board): return [i for i, c in enumerate(board) if c == " "]

# -------- Load Q-table --------
with open("qtable.pkl", "rb") as f:
    Q = pickle.load(f)

def choose_action(state, moves):
    # EVALUATION ONLY: no exploration
    if state not in Q:
        return random.choice(moves)
    return max(moves, key=lambda a: Q[state].get(a, 0))

# -------- Test Agent --------
games = 10000
wins = losses = draws = 0

for _ in range(games):
    board = initialize_board()
    done = False

    while not done:
        # Agent's turn
        moves = available_moves(board)
        action = choose_action(get_state(board), moves)
        board[action] = "X"

        winner = check_winner(board)
        if winner == "X":
            wins += 1
            done = True
            break
        elif is_full(board):
            draws += 1
            done = True
            break

        # Opponent (random)
        opp_moves = available_moves(board)
        opp_action = random.choice(opp_moves)
        board[opp_action] = "O"

        winner = check_winner(board)
        if winner == "O":
            losses += 1
            done = True
            break
        elif is_full(board):
            draws += 1
            done = True
            break

print(f"Results after {games} games vs Random Opponent:")
print(f"Wins:   {wins}")
print(f"Losses: {losses}")
print(f"Draws:  {draws}")

