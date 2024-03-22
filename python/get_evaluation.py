import chess

def get_evaluation(board):
  engine = chess.engine.SimpleEngine.configure({"Skill Level": 20})  
  result = engine.play(board, chess.engine.LimitTime(time=0.1))  
  evaluation = engine.evaluate(board)

  # Convert centipawn evaluation to a more user-friendly display string
  score_str = f"{evaluation/100:.2f}"  

  # Add symbol based on evaluation (positive for white advantage, negative for black)
  if evaluation > 0:
      score_str = "+" + score_str
  elif evaluation < 0:
      score_str = "-" + score_str

  return score_str
