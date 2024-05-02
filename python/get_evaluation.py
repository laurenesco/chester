from stockfish import Stockfish

def get_board_evaluation(fen_string):

  # Set up Stockfish engine (replace with your engine path if needed)
  engine = Stockfish(r"C:\Users\laesc\OneDrive\Desktop\chester\stockfish\stockfish-windows-x86-64-avx2.exe")

  engine.set_fen_position(fen_string)
  evaluation = engine.get_evaluation()

  if evaluation["type"] == "cp":
    return f"{evaluation['value'] / 100}?{evaluation['type']}"
  elif evaluation["type"] == "mate":
    return f"{evaluation['value']}?{evaluation['type']}"
  else:
    return "Evaluation type not recognized"

fen = "r1bqk1nr/pppp1ppp/n7/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1" 
print(get_board_evaluation(fen))
