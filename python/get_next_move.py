from stockfish import Stockfish

def get_next_best_move(uci_string):

  # Set up Stockfish engine (replace with your engine path if needed)
  engine = Stockfish(r"C:\Users\laesc\OneDrive\Desktop\chester\stockfish\stockfish-windows-x86-64-avx2.exe")
  engine.set_fen_position(uci_string)

  return engine.get_best_move()
