#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

struct evaluation {
	double value; // Represents advantage if cp, or moves until mate if mate
	int winning;  // 1 - white, 2 - black
	int status;   // 1 - cp, 2 - mate
};

int main(void) {

	evaluation eval;
	std::string eval_string = "-1.00?mate";

	int separator = eval_string.find('?');

	// Get the game state
	eval.status = (eval_string.substr(separator + 1) == "cp") ? 1 : 2;

   // Get the evaluation value
   std::stringstream(eval_string.substr(0, separator)) >> eval.value;

	// Determine who is winning
	eval.winning = eval.value >= 0 ? 1 : 2;

	// After determining who is winning, now can take abs value of value
   eval.value = std::abs(eval.value);

	std::cout << "value: " << eval.value << std::endl;
	std::cout << "status (1 - cp, 2 - mate): " << eval.status << std::endl;
   std::cout << "winning: " << eval.winning << std::endl;

	return 0;
}
