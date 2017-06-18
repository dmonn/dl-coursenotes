# C++ Quick Reference

## Basics

### Structure Hello World

```
include <iostream>
using namespace std; /* Don't have to type std:: before everything */

int main () {
	cout << "Hello";
	return 0;
}
```

### Output Formatting

```
std::cout<<"\nThe text with setw(15)\n";
std::cout<<"Ints" << std::setw(15)<<"Floats" << std::setw(15)<<"Doubles"<< "\n";
```

### File IO

- Include the <fstream> library 
- Create a stream (input, output, both)
  - ofstream myfile; (for writing to a file)
  - ifstream myfile; (for reading a file)
  - fstream myfile; (for reading and writing a file)
- Open the file  myfile.open(“filename”);
- Write or read the file
- Close the file myfile.close();

```
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main () {
    string line;
    //create an output stream to write to the file
    //append the new lines to the end of the file
    ofstream myfileI ("input.txt", ios::app);
    if (myfileI.is_open())
    {
        myfileI << "\nI am adding a line.\n";
        myfileI << "I am adding another line.\n";
        myfileI.close();
    }
    else cout << "Unable to open file for writing";

    //create an input stream to read the file
    ifstream myfileO ("input.txt");
    //During the creation of ifstream, the file is opened. 
    //So we do not have explicitly open the file. 
    if (myfileO.is_open())
    {
        while ( getline (myfileO,line) )
        {
            cout << line << '\n';
        }
        myfileO.close();
    }

    else cout << "Unable to open file for reading";

    return 0;
}
```

### Header Files

Header files contain information about how to do a task.

```
Main.cpp
#include "main.hpp"

int main()
{
    cout<<"Hello, I use header files!";
    return 0;
}


Main.hpp


#include <iostream>
#include <string>

using namespace std;

```

### User Input

```
Use std::cin >> age; for input (no spaces)

or

std::getline(std::cin, age)
```

## Pointers

Pointers, which are the addresses of variables, can be accessed in C++.

```
 // this is an integer variable with value = 54
int a = 54; 

// this is a pointer that holds the address of the variable 'a'.
// if 'a' was a float, rather than int, so should be its pointer.
int * pointerToA = &a;  

// If we were to print pointerToA, we'd obtain the address of 'a':
std::cout << "pointerToA stores " << pointerToA << '\n';

// If we want to know what is stored in this address, we can dereference pointerToA:
std::cout << "pointerToA points to " << * pointerToA << '\n';
```

## Arrays

Arrays can be declared as follows:

```
variableType arrayName [ ] = {variables to be stored in the array};

variableType arrayName[array size]

/* Access */
variableType arrayName[ index number ]
```

### Multidimensional

Multidimensional Arrays are declared as follows:

```
typeOfVariable arrayName[size of dim.1][size of dim. 2] ...[size of dim. n];

/* e.g. */
int array2Dimensions[2][3];
```
