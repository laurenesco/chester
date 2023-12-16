#include <iostream>
#include <boost/python.hpp>

using namespace boost::python;

int main() {

	try {
		// Initialize python interpretor
		Py_Initialize();

		// Import the module (our script)
		object main_module = import("__main__");
		object main_namespace = main_module.attr("__dict__");
		object my_module = exec_file("py_testing.py", main_namespace, main_namespace);

		// Get the functions from the module
		object return_value = main_namespace["return_value"];

		int result = etract<int>(return_value);

		std::cout << "result: " << result << std::end;

		// Finalize the interpretor
		Py_Finalize();
	} catch (const error_already_set&) {
		PyErr_Print();
	}

	return 0;

}
