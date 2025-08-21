import random
import pickle

# -------------------------
# Parameters
# -------------------------
alpha = 0.3           # learning rate
gamma = 0.9           # discount factor
epsilon = 0.7         # start high for exploration
epsilon_min = 0.01
decay = 0.9997        # decay per episode

# -------------------------
# Load or initialize Q-table
# -------------------------
try:
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
    print(f"✓ Loaded Q-table with {len(Q)} states")
except:
    Q = {}

# -------------------------
# Utility functions
# -------------------------
def initialize_board(): return [" "] * 9
def available_moves(board): return [i for i, v in enumerate(board) if v == " "]
def check_winner(board):
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != " ": return board[a]
    if " " not in board: return "D"  # Draw
    return None
def get_state(board, role): return "".join(board) + role

# -------------------------
# Minimax (for Smart opponent)
# -------------------------
def minimax(board, is_max, depth=0):
    winner = check_winner(board)
    if winner == "O": return 1 - 0.01*depth
    if winner == "X": return -1 + 0.01*depth
    if winner == "D": return 0
    if is_max:
        best = -999
        for m in available_moves(board):
            board[m] = "O"
            best = max(best, minimax(board, False, depth+1))
            board[m] = " "
        return best
    else:
        best = 999
        for m in available_moves(board):
            board[m] = "X"
            best = min(best, minimax(board, True, depth+1))
            board[m] = " "
        return best

def optimal_move(board):
    best_score = -999
    move = None
    for m in available_moves(board):
        board[m] = "O"
        score = minimax(board, False)
        board[m] = " "
        if score > best_score:
            best_score = score
            move = m
    return move

# -------------------------
# Q-learning functions
# -------------------------
def get_q(state, action):
    return Q.get(state, {}).get(action, 0.0)

def choose_action(state, moves):
    if random.random() < epsilon:
        return random.choice(moves)
    q_vals = [(m, get_q(state, m)) for m in moves]
    max_q = max(q_vals, key=lambda x:x[1])[1]
    best = [m for m, q in q_vals if q==max_q]
    return random.choice(best)

def update_q(state, action, reward, next_state, next_moves):
    if state not in Q: Q[state] = {}
    current = Q[state].get(action, 0.0)
    next_max = max([get_q(next_state, m) for m in next_moves], default=0)
    Q[state][action] = current + alpha * (reward + gamma * next_max - current)

# -------------------------
# Training loop
# -------------------------
modes = [("random", 300_000), ("selfplay", 500_000), ("minimax", 5_000)]

for mode, episodes in modes:
    print(f"\nTraining mode: {mode.upper()} | Episodes: {episodes}")
    for ep in range(episodes):
        board = initialize_board()
        history = []  # store (state, action, role)
        done = False

        while not done:
            # X's turn
            state_X = get_state(board, "X")
            moves_X = available_moves(board)
            action_X = choose_action(state_X, moves_X)
            board[action_X] = "X"
            history.append((state_X, action_X, "X"))

            winner = check_winner(board)
            if winner:
                reward = 10 if winner=="X" else (-10 if winner=="O" else 0)
                for s,a,r in [(s,a,10 if r=="X" else -10 if r=="O" else 0) for s,a,r in history]:
                    update_q(s, a, reward, "", [])
                done = True
                break

            # O's turn
            state_O = get_state(board, "O")
            moves_O = available_moves(board)
            if mode=="random":
                action_O = random.choice(moves_O)
            elif mode=="minimax":
                action_O = optimal_move(board)
            else:  # selfplay
                action_O = choose_action(state_O, moves_O)

            board[action_O] = "O"
            history.append((state_O, action_O, "O"))

            winner = check_winner(board)
            if winner:
                reward = 10 if winner=="X" else (-10 if winner=="O" else 0)
                for s,a,r in [(s,a,10 if r=="X" else -10 if r=="O" else 0) for s,a,r in history]:
                    update_q(s, a, reward, "", [])
                done = True
                break

            # Small intermediate rewards
            next_state_X = get_state(board, "X")
            update_q(state_X, action_X, -0.01, next_state_X, available_moves(board))
            next_state_O = get_state(board, "O")
            update_q(state_O, action_O, -0.01, next_state_O, available_moves(board))

        # Decay epsilon
        epsilon
        if epsilon > epsilon_min: epsilon *= decay

        # Progress
        if (ep+1) % 5000 == 0:
            print(f"Ep {ep+1}/{episodes} | Q-size: {len(Q)} | ε={epsilon:.4f}")

# -------------------------
# Save Q-table
# -------------------------
with open("qtable.pkl", "wb") as f:
    pickle.dump(Q, f)

print("\nTRAINING COMPLETE")
print(f"Q-table size: {len(Q)} | Final ε={epsilon:.4f}")
