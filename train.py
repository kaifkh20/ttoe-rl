import random
import pickle

# ===============================
# Tic Tac Toe Environment
# ===============================

def initialize_board():
    return [" " for _ in range(9)]

def print_board(board):
    print("Current board:")
    for i in range(0, 9, 3):
        print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
        if i < 6:
            print("-----------")
    print()

def available_moves(board):
    return [i for i in range(9) if board[i] == " "]

def check_winner(board):
    win_states = [
        [0,1,2],[3,4,5],[6,7,8],   # rows
        [0,3,6],[1,4,7],[2,5,8],   # cols
        [0,4,8],[2,4,6]            # diagonals
    ]
    for a,b,c in win_states:
        if board[a] == board[b] == board[c] and board[a] != " ":
            return board[a]
    return None

def is_full(board):
    return " " not in board

def get_state(board):
    return "".join(board)

# ===============================
# Minimax for Smart Opponent
# ===============================

def minimax(board, is_maximizing, depth=0):
    winner = check_winner(board)
    if winner == "O": return 1 - depth * 0.01  # Prefer faster wins
    if winner == "X": return -1 + depth * 0.01  # Prefer slower losses
    if is_full(board): return 0  # Tie is neutral

    if is_maximizing:
        best = -999
        for move in available_moves(board):
            board[move] = "O"
            score = minimax(board, False, depth + 1)
            board[move] = " "
            best = max(best, score)
        return best
    else:
        best = 999
        for move in available_moves(board):
            board[move] = "X"
            score = minimax(board, True, depth + 1)
            board[move] = " "
            best = min(best, score)
        return best

def optimal_move(board):
    best_score = -999
    move = None
    available = available_moves(board)
    
    if not available:  # Safety check
        return None
        
    for m in available:
        board[m] = "O"
        score = minimax(board, False)
        board[m] = " "
        if score > best_score:
            best_score = score
            move = m
    return move

# ===============================
# Q-learning Agent
# ===============================


try :
    with open("qtable.pkl","rb"):
        Q = pickle.load(f);
except:
    Q= {}

alpha = 0.1      # Lower learning rate for more stable learning
gamma = 0.9      # Discount factor
epsilon = 0.3    # Start with lower exploration rate
epsilon_min = 0.01
decay = 0.9995   # Slower decay

def get_q_value(state, action):
    """Get Q-value for state-action pair"""
    return Q.get(state, {}).get(action, 0.0)

def choose_action(state, moves):
    """Choose action using epsilon-greedy policy"""
    if random.uniform(0, 1) < epsilon:
        return random.choice(moves)  # explore
    
    # Exploit: choose action with highest Q-value
    q_values = [(move, get_q_value(state, move)) for move in moves]
    max_q = max(q_values, key=lambda x: x[1])[1]
    best_moves = [move for move, q in q_values if q == max_q]
    return random.choice(best_moves)

def update_q_value(state, action, reward, next_state, next_moves):
    """Update Q-value using Q-learning formula"""
    if state not in Q:
        Q[state] = {}
    
    current_q = Q[state].get(action, 0.0)
    
    if next_moves:  # Game continues
        next_max_q = max([get_q_value(next_state, move) for move in next_moves])
    else:  # Terminal state
        next_max_q = 0.0
    
    # Q-learning update formula
    new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
    Q[state][action] = new_q

# ===============================
# Training Loop
# ===============================

use_smart_opponent = False  # Toggle between Random and Optimal opponent
episodes = 30000 if use_smart_opponent else 200000

print(f"Starting training for {episodes} episodes...")
print(f"Opponent type: {'Smart (Minimax)' if use_smart_opponent else 'Random'}")

wins = 0
losses = 0
draws = 0



for ep in range(episodes):
    board = initialize_board()
    game_history = []  # Store (state, action) pairs for this game
    done = False

    while not done:
        # Agent's turn (X)
        state = get_state(board)
        moves = available_moves(board)
        
        if not moves:  # Board is full
            done = True
            break
            
        action = choose_action(state, moves)
        game_history.append((state, action))
        
        # Make the move
        board[action] = "X"
        winner = check_winner(board)
        
        # Check if agent won
        if winner == "X":
            wins += 1
            # Update Q-values for all moves in this game
            for i, (s, a) in enumerate(game_history):
                # Give higher reward for moves closer to the end
                reward = 10 + (len(game_history) - i) * 0.1
                update_q_value(s, a, reward, "", [])
            done = True
            continue
        
        # Check if board is full (draw)
        if is_full(board):
            draws += 1
            # Update Q-values for draw
            for s, a in game_history:
                update_q_value(s, a, 1, "", [])  # Small positive reward for draw
            done = True
            continue
        
        # Opponent's turn (O)
        opp_moves = available_moves(board)
        if not opp_moves:
            done = True
            continue
            
        if use_smart_opponent:
            opp_action = optimal_move(board)
            if opp_action is None:  # Safety fallback
                opp_action = random.choice(opp_moves)
        else:
            opp_action = random.choice(opp_moves)
        
        board[opp_action] = "O"
        winner = check_winner(board)
        
        # Check if opponent won
        if winner == "O":
            losses += 1
            # Negative reward for losing
            for s, a in game_history:
                update_q_value(s, a, -10, "", [])
            done = True
            continue
        
        # Check if board is full after opponent's move
        if is_full(board):
            draws += 1
            # Small positive reward for draw
            for s, a in game_history:
                update_q_value(s, a, 1, "", [])
            done = True
            continue
        
        # Game continues - give small negative reward to encourage faster wins
        next_state = get_state(board)
        next_moves = available_moves(board)
        
        # Only update the last move with intermediate reward
        if game_history:
            last_state, last_action = game_history[-1]
            update_q_value(last_state, last_action, -0.01, next_state, next_moves)

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= decay

    # Print progress
    if (ep + 1) % 5000 == 0:
        total_games = ep + 1
        win_rate = wins / total_games * 100
        loss_rate = losses / total_games * 100
        draw_rate = draws / total_games * 100
        print(f"Episode {ep + 1}/{episodes}")
        print(f"  Win rate: {win_rate:.1f}% ({wins})")
        print(f"  Loss rate: {loss_rate:.1f}% ({losses})")
        print(f"  Draw rate: {draw_rate:.1f}% ({draws})")
        print(f"  Epsilon: {epsilon:.4f}")
        print(f"  Q-table size: {len(Q)}")
        print()

# Final statistics
total_games = episodes
win_rate = wins / total_games * 100
loss_rate = losses / total_games * 100
draw_rate = draws / total_games * 100

print("="*50)
print("TRAINING COMPLETED!")
print("="*50)
print(f"Total games: {total_games}")
print(f"Final win rate: {win_rate:.2f}% ({wins})")
print(f"Final loss rate: {loss_rate:.2f}% ({losses})")
print(f"Final draw rate: {draw_rate:.2f}% ({draws})")
print(f"Final epsilon: {epsilon:.4f}")
print(f"Q-table states learned: {len(Q)}")
print()

# ===============================
# Save Trained Q-table
# ===============================

try:
    with open("qtable.pkl", "wb") as f:
        pickle.dump(Q, f)
    print("✓ Training completed successfully!")
    print("✓ Q-table saved as 'qtable.pkl'")
except Exception as e:
    print(f"Error saving Q-table: {e}")
