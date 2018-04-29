#include <iostream>

#include <fstream>

#include <string>

#include <vector>

#include <algorithm>    // std::shuffle

#include <array>        // std::array

#include <random>       // std::default_random_engine

#include <chrono>       // std::chrono::system_clock
#define USER_SIZE 458293
#define MOVIE_SIZE 17700

//change these parameters for each file
int read_in_lines 94362234;
string filename = "all.idx";
int parameters_per_line = 4
using namespace std;



int main() {

	ifstream inFile;
	inFile.open(filename);
	cout << "Reading data from " << filename << endl;

	while (!inFile.eof()) {

		int x;
		for (int i = 0; i < read_in_lines; i++)
		{
			for (int i = 0; i < parameters_per_line; i++)
			{
				inFile >> x;
				if (x <= 0 || x < USER_SIZE)
				{
					cout << "Entry " << x << " on line " << "i " << "is an error" << endl;
					exit(1);
				}
			}
		}


	}

	inFile.close();
	cout << "FILE : " << filename << " ES GOOD"<< endl;
	cout << endl;
	return 0;

}
