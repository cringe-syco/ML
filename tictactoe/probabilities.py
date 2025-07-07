# probabilities.py

def minimax_probabilities(board, turn):
    def check_winner(b):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for a,b_,c in wins:
            if b[a] and b[a] == b[b_] == b[c]:
                return b[a]
        if all(b):
            return "Draw"
        return None
    def simulate(b, t):
        winner = check_winner(b)
        if winner == "X": return (1,0,0)
        if winner == "O": return (0,1,0)
        if winner == "Draw": return (0,0,1)
        x = o = d = 0
        for i in range(9):
            if b[i] == "":
                new_b = b[:]
                new_b[i] = t
                next_t = "O" if t == "X" else "X"
                sx, so, sd = simulate(new_b, next_t)
                x += sx
                o += so
                d += sd
        return (x, o, d)
    x, o, d = simulate(board, turn)
    total = x + o + d
    if total == 0:
        return 50
    return int(round(100 * o / total))
