#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

int main() {

  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
