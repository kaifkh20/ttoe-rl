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

def is_full(board): 
    return " " not in board

def get_state(board): 
    return "".join(board)

def available_moves(board): 
    return [i for i, c in enumerate(board) if c == " "]

# -------- Minimax for Smart Opponent --------
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

# -------- Load Q-table --------
try:
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
    print(f"✓ Loaded Q-table with {len(Q)} states")
except FileNotFoundError:
    print("Error: qtable.pkl not found. Please train the agent first.")
    exit()

def choose_action(state, moves):
    # EVALUATION ONLY: no exploration
    if state not in Q:
        return random.choice(moves)
    return max(moves, key=lambda a: Q[state].get(a, 0))

# -------- Test Function --------
def test_agent(opponent_type, games=100000):
    wins = losses = draws = 0
    
    print(f"Testing against {opponent_type} opponent ({games:,} games)...")
    
    for game in range(games):
        board = initialize_board()
        done = False
        
        # Progress indicator for long tests
        if (game + 1) % 10000 == 0:
            print(f"  Progress: {game + 1:,}/{games:,} games")
        
        while not done:
            # Agent's turn (X)
            moves = available_moves(board)
            if not moves:
                break
                
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
            
            # Opponent's turn (O)
            opp_moves = available_moves(board)
            if not opp_moves:
                break
                
            if opponent_type == "Smart":
                opp_action = optimal_move(board)
                if opp_action is None:  # Safety fallback
                    opp_action = random.choice(opp_moves)
            else:  # Random opponent
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
    
    return wins, losses, draws

# -------- Run Tests --------
def main():
    test_games = 10000 # Reduced for faster testing, increase if needed
    
    print("=" * 60)
    print("Q-LEARNING AGENT EVALUATION")
    print("=" * 60)
    
    # Test against Random opponent
    print("\n1. Testing against RANDOM opponent:")
    wins_r, losses_r, draws_r = test_agent("Random", test_games)
    
    win_rate_r = wins_r / test_games * 100
    loss_rate_r = losses_r / test_games * 100
    draw_rate_r = draws_r / test_games * 100
    
    print(f"\nResults vs Random Opponent ({test_games:,} games):")
    print(f"  Wins:   {wins_r:,} ({win_rate_r:.2f}%)")
    print(f"  Losses: {losses_r:,} ({loss_rate_r:.2f}%)")
    print(f"  Draws:  {draws_r:,} ({draw_rate_r:.2f}%)")
    
    # Test against Smart opponent
    print("\n" + "=" * 60)
    print("\n2. Testing against SMART (Minimax) opponent:")
    wins_s, losses_s, draws_s = test_agent("Smart", test_games)
    
    win_rate_s = wins_s / test_games * 100
    loss_rate_s = losses_s / test_games * 100
    draw_rate_s = draws_s / test_games * 100
    
    print(f"\nResults vs Smart Opponent ({test_games:,} games):")
    print(f"  Wins:   {wins_s:,} ({win_rate_s:.2f}%)")
    print(f"  Losses: {losses_s:,} ({loss_rate_s:.2f}%)")
    print(f"  Draws:  {draws_s:,} ({draw_rate_s:.2f}%)")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Opponent':<12} {'Win Rate':<12} {'Loss Rate':<12} {'Draw Rate':<12}")
    print("-" * 48)
    print(f"{'Random':<12} {win_rate_r:<11.2f}% {loss_rate_r:<11.2f}% {draw_rate_r:<11.2f}%")
    print(f"{'Smart':<12} {win_rate_s:<11.2f}% {loss_rate_s:<11.2f}% {draw_rate_s:<11.2f}%")
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if win_rate_r > 90:
        print("✓ Excellent performance vs Random opponent!")
    elif win_rate_r > 70:
        print("✓ Good performance vs Random opponent")
    else:
        print("⚠ Poor performance vs Random opponent - may need more training")
    
    if win_rate_s > 10:
        print("✓ Impressive! Agent can beat optimal play sometimes")
    elif draw_rate_s > 80:
        print("✓ Excellent! Agent draws most games vs optimal opponent")
    elif draw_rate_s > 50:
        print("✓ Good! Agent avoids losing to optimal opponent")
    else:
        print("⚠ Agent struggles against optimal play")
    
    print(f"\nOverall: Agent learned to play at {'Expert' if draw_rate_s > 80 else 'Intermediate' if draw_rate_s > 50 else 'Beginner'} level")

if __name__ == "__main__":
    main()
