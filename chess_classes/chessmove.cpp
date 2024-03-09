#include "chessmove.h"

ChessMove::ChessMove(){
}

// This function is reponsible for ensuring that...
//      - the board is valid (two kings on board?)
//      - the move was legal for that piece
//      - any new checks are notated
//      - any captures are completed
//      - en passant status is notated
//      - castling rights are updated
//      - any consecutive repeat moves are notated
//      - stalemates are caught

