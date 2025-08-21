import tkinter as tk
import random
import pickle

# ---------- Q-learning parameters ----------
alpha = 0.5    # learning rate
gamma = 0.9    # discount factor

# ---------- Load or init Q-table ----------
try:
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
    print(f"✓ Loaded Q-table with {len(Q)} states")
except FileNotFoundError:
    Q = {}
    print("⚠️ No Q-table found, starting fresh.")

def convert_to_training_format(board):
    """Convert GUI board format to training format"""
    mapping = {0: " ", 1: "X", -1: "O"}
    return [mapping[cell] for cell in board]

def get_state(board):
    return "".join(convert_to_training_format(board))

epsilon = 0.2  # 20% chance to explore instead of exploit

def best_ai_move(board):
    available_moves = [i for i in range(9) if board[i] == 0]
    if not available_moves:
        return None
    
    state = get_state(board)
    if state not in Q:
        Q[state] = {}
    
    # 🔥 With probability ε, pick a random move (explore)
    if random.random() < epsilon:
        return random.choice(available_moves)
    
    # Otherwise exploit best known move
    q_values = [(move, Q[state].get(move, 0.0)) for move in available_moves]
    max_q = max(q_values, key=lambda x: x[1])[1]
    best_moves = [m for m, q in q_values if q == max_q]
    
    return random.choice(best_moves)

def update_q(prev_state, action, reward, next_state):
    """Update Q-table with Q-learning formula"""
    if prev_state not in Q:
        Q[prev_state] = {}
    if action not in Q[prev_state]:
        Q[prev_state][action] = 0.0
    if next_state not in Q:
        Q[next_state] = {}
    
    old_q = Q[prev_state][action]
    future_q = max(Q[next_state].values(), default=0.0)
    Q[prev_state][action] = old_q + alpha * (reward + gamma * future_q - old_q)

# ---------- Game logic ----------
board = [0] * 9
buttons = []
game_over = False
ai_memory = []  # store (state, action) for this game

def check_winner(board):
    win_patterns = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0
    return None

def handle_click(i):
    global game_over
    
    if board[i] != 0 or game_over:
        return
    
    # Human move
    board[i] = -1
    buttons[i].config(text="O", bg="lightblue", state="disabled")
    
    result = check_winner(board)
    if result is not None:
        end_game(result)
        return
    
    # AI move
    state = get_state(board)
    ai_move = best_ai_move(board)
    if ai_move is not None:
        ai_memory.append((state, ai_move))  # remember choice
        board[ai_move] = 1
        buttons[ai_move].config(text="X", bg="lightcoral", state="disabled")
        
        result = check_winner(board)
        if result is not None:
            end_game(result)
            return
    
    status.config(text="Your turn (O)!")

def end_game(result):
    global game_over
    game_over = True
    
    # Assign rewards
    if result == 1:
        reward = 1   # AI win
        status.config(text="AI wins! 🤖", fg="red")
    elif result == -1:
        reward = -1  # Human win
        status.config(text="You win! 🎉", fg="green")
    else:
        reward = 0   # Draw
        status.config(text="It's a draw! 🤝", fg="blue")
    
    # Train from memory
    next_state = get_state(board)
    for state, action in reversed(ai_memory):
        update_q(state, action, reward, next_state)
        next_state = state
        reward = 0  # only final outcome gets nonzero reward
    
    # Save Q-table
    with open("qtable.pkl", "wb") as f:
        pickle.dump(Q, f)
    
    # Disable all buttons
    for b in buttons:
        b.config(state="disabled")

def reset_game():
    global board, game_over, ai_memory
    board = [0]*9
    game_over = False
    ai_memory = []
    for b in buttons:
        b.config(text=" ", bg="gray90", state="normal")
    status.config(text="Your turn (O)! Click any square.", fg="black")

# ---------- GUI ----------
root = tk.Tk()
root.title("Tic Tac Toe vs Q-Learning AI")
root.geometry("400x500")
root.resizable(False, False)

title_label = tk.Label(root, text="Tic Tac Toe", font=("Arial", 20, "bold"))
title_label.pack(pady=10)
subtitle_label = tk.Label(root, text="You are O, AI is X", font=("Arial", 12))
subtitle_label.pack()

board_frame = tk.Frame(root, bg="black")
board_frame.pack(pady=20)

for i in range(9):
    row, col = divmod(i,3)
    btn = tk.Button(board_frame, text=" ", font=("Arial", 24, "bold"),
                    width=4, height=2, command=lambda i=i: handle_click(i))
    btn.grid(row=row, column=col, padx=2, pady=2)
    buttons.append(btn)

status = tk.Label(root, text="Your turn (O)! Click any square.", font=("Arial", 14))
status.pack(pady=10)

control_frame = tk.Frame(root)
control_frame.pack(pady=10)

tk.Button(control_frame, text="New Game", font=("Arial", 12),
          command=reset_game, bg="lightgreen", padx=20).pack(side=tk.LEFT, padx=10)
tk.Button(control_frame, text="Quit", font=("Arial", 12),
          command=root.quit, bg="lightcoral", padx=20).pack(side=tk.LEFT, padx=10)

info_label = tk.Label(root, text="AI learns with Q-Learning while you play",
                      font=("Arial", 10), fg="gray")
info_label.pack(side=tk.BOTTOM, pady=5)

reset_game()
root.mainloop()
