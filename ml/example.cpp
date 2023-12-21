#include <iostream>
#include <boost/python.hpp>

void helloWorld()
{
    std::cout << "Hello" << std::endl;
}

BOOST_PYTHON_MODULE(hello)
{
    using namespace boost::python;
    def("world", &helloWorld);
}
