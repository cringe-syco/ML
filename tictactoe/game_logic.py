# game_logic.py

class Game:
    def __init__(self, prob_calc=None):
        self.prob_calc = prob_calc
        self.reset()
    def reset(self):
        self.board = ["" for _ in range(9)]
        self.turn = "X"
        self.game_over = False
        self.status = "Player X's turn"
        self.prob_o = 50
    def move(self, pos):
        if self.game_over or self.board[pos] != "":
            return
        self.board[pos] = self.turn
        winner = self.check_winner()
        if winner:
            self.game_over = True
            if winner == "Draw":
                self.status = "It's a draw!"
            else:
                self.status = f"{winner} wins!"
        else:
            self.turn = "O" if self.turn == "X" else "X"
            self.status = f"Player {self.turn}'s turn"
        if self.prob_calc:
            self.prob_o = self.prob_calc(self.board, self.turn)
        else:
            self.prob_o = 50
    def check_winner(self):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for a,b,c in wins:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if all(self.board):
            return "Draw"
        return None
    def state(self):
        return {
            "board": self.board,
            "turn": self.turn,
            "game_over": self.game_over,
            "status": self.status,
            "prob_o": self.prob_o
        }
