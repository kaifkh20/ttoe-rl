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
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] != " ":
            return board[a]
    return None

def is_full(board):
    return " " not in board

def available_moves(board):
    return [i for i, c in enumerate(board) if c == " "]

# -------- Minimax for Smart Opponent --------
def minimax(board, is_maximizing, depth=0):
    winner = check_winner(board)
    if winner == "O": return 1 - depth * 0.01
    if winner == "X": return -1 + depth * 0.01
    if is_full(board): return 0

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
    best = None
    for m in available_moves(board):
        board[m] = "O"
        score = minimax(board, False)
        board[m] = " "
        if score > best_score:
            best_score = score
            best = m
    return best

# -------- Load Q-table --------
try:
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
    print(f"✓ Loaded Q-table with {len(Q)} states")
except FileNotFoundError:
    print("Error: qtable.pkl not found. Please train the agent first.")
    raise SystemExit

# --- Auto-detect whether Q-table is role-aware (board + 'X'/'O') ---
def is_role_aware(qtable) -> bool:
    try:
        k = next(iter(qtable.keys()))
    except StopIteration:
        return True  # assume role-aware if empty
    return len(k) == 10  # 9 board chars + 1 role

ROLE_AWARE = is_role_aware(Q)

def get_state(board, role="X"):
    s = "".join(board)
    return s + role if ROLE_AWARE else s

def choose_action(state, moves):
    # Evaluation-only: greedy
    if state not in Q:
        return random.choice(moves)
    # Use 0.0 for unseen actions in this state
    return max(moves, key=lambda a: Q[state].get(a, 0.0))

# -------- Test Function --------
def test_agent(opponent_type: str, n_games: int):
    """
    Runs n_games and returns (wins, losses, draws, n_games_used).
    If n_games is None, uses sensible defaults per opponent.
    """
    defaults = {"Random": 100_000, "Smart":1}
    if n_games is None:
        n_games = defaults.get(opponent_type, 10_000)

    wins = losses = draws = 0
    print(f"Testing against {opponent_type} opponent ({n_games:,} games)...")

    # Print progress ~10 times (but not too chatty for small runs)
    step = max(1, n_games // 10)

    for game in range(n_games):
        board = initialize_board()
        done = False

        if (game + 1) % step == 0:
            print(f"  Progress: {game + 1:,}/{n_games:,}")

        while not done:
            # Agent (X)
            moves = available_moves(board)
            if not moves:
                break

            action = choose_action(get_state(board, "X"), moves)
            board[action] = "X"
            winner = check_winner(board)

            if winner == "X":
                wins += 1
                done = True
                break
            if is_full(board):
                draws += 1
                done = True
                break

            # Opponent (O)
            opp_moves = available_moves(board)
            if not opp_moves:
                break

            if opponent_type == "Smart":
                opp_action = optimal_move(board)
                if opp_action is None:
                    opp_action = random.choice(opp_moves)
            else:
                opp_action = random.choice(opp_moves)

            board[opp_action] = "O"
            winner = check_winner(board)

            if winner == "O":
                losses += 1
                done = True
                break
            if is_full(board):
                draws += 1
                done = True
                break

    return wins, losses, draws, n_games

# -------- Exploration Progress (board-pattern aware) --------
def exploration_progress():
    total_legal_boards = 5478
    total_states = len(Q)
    # Collapse role-aware keys to board-only patterns
    unique_boards = len({k[:9] for k in Q.keys()})
    progress = (unique_boards / total_legal_boards) * 100
    print("\nState-space coverage:")
    print(f"  Q entries (role-aware states): {total_states:,}")
    print(f"  Unique board patterns:        {unique_boards:,}/{total_legal_boards} ({progress:.2f}%)")

# -------- Run Tests --------
def main():
    # You can set your own counts; if None, defaults are used inside test_agent.
    random_games = None   # e.g., 100_000
    smart_games  = None   # e.g., 5_000

    print("=" * 60)
    print("Q-LEARNING AGENT EVALUATION")
    print("=" * 60)

    # 1) Random opponent
    print("\n1. Testing against RANDOM opponent:")
    wins_r, losses_r, draws_r, n_r = test_agent("Random", random_games)

    win_rate_r  = wins_r  / n_r * 100
    loss_rate_r = losses_r / n_r * 100
    draw_rate_r = draws_r / n_r * 100

    print(f"\nResults vs Random Opponent ({n_r:,} games):")
    print(f"  Wins:   {wins_r:,} ({win_rate_r:.2f}%)")
    print(f"  Losses: {losses_r:,} ({loss_rate_r:.2f}%)")
    print(f"  Draws:  {draws_r:,} ({draw_rate_r:.2f}%)")

    # 2) Smart (minimax) opponent
    print("\n" + "=" * 60)
    print("\n2. Testing against SMART (Minimax) opponent:")
    wins_s, losses_s, draws_s, n_s = test_agent("Smart", smart_games)

    win_rate_s  = wins_s  / n_s * 100
    loss_rate_s = losses_s / n_s * 100
    draw_rate_s = draws_s / n_s * 100

    print(f"\nResults vs Smart Opponent ({n_s:,} games):")
    print(f"  Wins:   {wins_s:,} ({win_rate_s:.2f}%)")
    print(f"  Losses: {losses_s:,} ({loss_rate_s:.2f}%)")
    print(f"  Draws:  {draws_s:,} ({draw_rate_s:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Opponent':<12} {'Win Rate':<12} {'Loss Rate':<12} {'Draw Rate':<12}")
    print("-" * 48)
    print(f"{'Random':<12} {win_rate_r:<11.2f}% {loss_rate_r:<11.2f}% {draw_rate_r:<11.2f}%")
    print(f"{'Smart':<12} {win_rate_s:<11.2f}% {loss_rate_s:<11.2f}% {draw_rate_s:<11.2f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    if win_rate_r > 90:
        print("✓ Excellent vs Random (great at exploiting mistakes).")
    elif win_rate_r > 70:
        print("✓ Good vs Random.")
    else:
        print("⚠ Needs more training/exploration vs Random.")

    if win_rate_s > 0:
        print("✓ Any wins vs Minimax are rare — interesting!")
    if draw_rate_s > 80:
        print("✓ Excellent: mostly draws vs optimal opponent.")
    elif draw_rate_s > 50:
        print("✓ Decent: avoids many losses vs optimal opponent.")
    else:
        print("⚠ Struggles vs optimal play — keep training (self-play helps).")

    level = 'Expert' if draw_rate_s > 80 else 'Intermediate' if draw_rate_s > 50 else 'Beginner'
    print(f"\nOverall Level vs Optimal: {level}")

    exploration_progress()

if __name__ == "__main__":
    main()
