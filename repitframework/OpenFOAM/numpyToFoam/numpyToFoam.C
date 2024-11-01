#include <iostream>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
using namespace std;

void init_numpy() {
  import_array1();
}

double square (double x ) {
    return x * x;
}

void print_square(double x) {
    cout << "The square of " << x << " is " << square(x) << "\n";
}

int main() {
    std::cout << "Header linkage successful!" << std::endl;
    print_square(2);
    return 0;
}