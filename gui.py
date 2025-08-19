import tkinter as tk, random, pickle

# ---------- Load Q-table ----------
with open("qtable.pkl","rb") as f:
    Q = pickle.load(f)

def get_q(board, action):
    return Q.get((tuple(board), action), 0.0)

def best_ai_move(board):
    moves = [i for i in range(9) if board[i] == 0]
    if not moves: return None
    qs = [get_q(board,a) for a in moves]
    best = max(qs, default=0)
    best_moves = [a for a in moves if get_q(board,a) == best]
    return random.choice(best_moves)

# ---------- Game logic ----------
board = [0]*9  # 0 empty, 1 AI, -1 Human
buttons = []

def check_winner(b):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,b,c in wins:
        s = board[a]+board[b]+board[c]
        if s==3: return 1
        if s==-3: return -1
    if 0 not in board: return 0
    return None

def handle_click(i):
    if board[i] != 0: return
    board[i] = -1
    buttons[i].config(text="O", state="disabled")
    result = check_winner(board)
    if result is not None: end_game(result); return

    ai = best_ai_move(board)
    if ai is not None:
        board[ai] = 1
        buttons[ai].config(text="X", state="disabled")
    result = check_winner(board)
    if result is not None: end_game(result)

def end_game(result):
    if result == 1: msg = "AI wins!"
    elif result == -1: msg = "You win!"
    else: msg = "Draw."
    status.config(text=msg)
    for b in buttons: b.config(state="disabled")

def reset_game():
    global board
    board = [0]*9
    for b in buttons: b.config(text=" ", state="normal")
    status.config(text="Your turn (O)!")

# ---------- Tkinter GUI ----------
root = tk.Tk()
root.title("Tic Tac Toe RL")

frame = tk.Frame(root)
frame.pack()

for i in range(9):
    b = tk.Button(frame, text=" ", font=("Arial", 24), width=5, height=2,
                  command=lambda i=i: handle_click(i))
    b.grid(row=i//3, column=i%3)
    buttons.append(b)

status = tk.Label(root, text="Your turn (O)!", font=("Arial", 14))
status.pack()

reset_btn = tk.Button(root, text="Reset", command=reset_game)
reset_btn.pack()

root.mainloop()

