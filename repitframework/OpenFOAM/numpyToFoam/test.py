print("Hello Bro!")#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <string>


int main(int argc, char *argv[])
{
    // Python C/API initialization
    Py_Initialize();

    // import test
    printf("Importing test module\n");
    PyObject *pNumpy = PyUnicode_FromString("numpy");
    if(!pNumpy)
    {
        PyErr_Print();
        printf("Failed to create Python string for module name 'numpy'\n");
        Py_Finalize();
        return 1;
    }

    PyObject *numpy = PyImport_Import(pNumpy);
    Py_DECREF(pNumpy);
    if(!numpy)
    {
        PyErr_Print();
        printf("Error importing module\n");
        Py_Finalize();
        return 1;
    }

    // Get the numpy load function:
    PyObject *pNumpyLoad = PyObject_GetAttrString(numpy, "load");
    if (!pNumpyLoad || !PyCallable_Check(pNumpyLoad))
    {
        if (PyErr_Occurred())
            PyErr_Print();
        printf("Error getting numpy.load\n");
        Py_DECREF(numpy);
        Py_Finalize();
        return 1;
    }

    // Load the .npz file
    const char *filename = argv[1];
    PyObject *pArgs = PyTuple_Pack(1, PyUnicode_FromString(filename));
    if (!pArgs)
    {
        PyErr_Print();
        printf("Error creating arguments\n");
        Py_DECREF(pNumpyLoad);
        Py_DECREF(numpy);
        Py_Finalize();
        return 1;
    }

    PyObject *pData = PyObject_CallObject(pNumpyLoad, pArgs);
    Py_DECREF(pArgs);
    if (!pData)
    {
        PyErr_Print();
        printf("Error loading data\n");
        Py_DECREF(pNumpyLoad);
        Py_DECREF(numpy);
        Py_Finalize();
        return 1;
    }

    // Get the shape of the array
    PyObject *pShape = PyObject_GetAttrString(pData, "shape");
    if (!pShape)
    {
        PyErr_Print();
        printf("Error getting shape\n");
        Py_DECREF(pData);
        Py_Finalize();
        return 1;
    }

    // Convert the shape to a string for printing
    PyObject *pShapeStr = PyObject_Str(pShape);
    if (!pShapeStr)
    {
        PyErr_Print();
        printf("Error converting shape to string\n");
        Py_DECREF(pShape);
        Py_DECREF(pData);
        Py_Finalize();
        return 1;
    }

    // Print the shape
    printf("Shape: %s\n", PyUnicode_AsUTF8(pShapeStr));
    
    // Clean up
    Py_DECREF(pShapeStr);
    Py_DECREF(pShape);
    Py_DECREF(pData);

    return 0;


}


// ************************************************************************* //