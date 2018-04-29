#include <iostream>

#include <fstream>

#include <string>

#include <vector>

#include <algorithm>    // std::shuffle

#include <array>        // std::array

#include <random>       // std::default_random_engine

#include <chrono>       // std::chrono::system_clock
#define USER_SIZE 458293
#define READ_IN_LINES 94362234
#define MOVIE_SIZE 17700
#define FILENAME "all.idx"
#define PARAMETERS_PER_LINE 4



using namespace std;



int shuffle() {

	vector <int> list_of_indices;
	ifstream inFile;
	inFile.open(FILENAME);

	unsigned long long count = 0;
	long long count1 = 0;
	cout << "Reading data from all.idx" << endl;

	while (!inFile.eof()) {

		int x;
		for (int i = 0; i < PARAMETERS_PER_LINE; i++)
		{
			inFile >> x;
			if (x <= 0 || x < USER_SIZE)
			{
				cout << "Entry " << x << " on line " << "i " << "is an error" << endl;
				exit(1);
			}
		}
			


	}

	inFile.close();
	cout << "FILE : " << FILENAME<< " ES GOOD"<< endl;
	cout << endl;
	return 0;

}