from stockfish import Stockfish

def get_game_stats(fen_string):
  engine = Stockfish(r"C:\Users\laesc\OneDrive\Desktop\chester\stockfish\stockfish-windows-x86-64-avx2.exe")
  engine.set_depth(5)
  engine.set_elo_rating(800)

  engine.set_fen_position(fen_string)

  move = engine.get_best_move()

  evaluation = engine.get_evaluation()

  if evaluation["type"] == "cp":
    game_eval = f"{evaluation['value'] / 100}?{evaluation['type']}"
  elif evaluation["type"] == "mate":
    game_eval = f"{evaluation['value']}?{evaluation['type']}"

  return f"{move}&{game_eval}"
