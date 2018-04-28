#include <iostream>

#include <fstream>

#include <string>

#include <vector>

#include <algorithm>    // std::shuffle

#include <array>        // std::array

#include <random>       // std::default_random_engine

#include <chrono>       // std::chrono::system_clock
#define USER_SIZE 458294
#define READ_IN_LINES 102416306
#define MOVIE_SIZE 17701


#define TRAIN_SIZE  102416306

#define DATASET_NUM  1  // The dataset that we want to select



using namespace std;



int shuffle () {

	vector <int> list_of_indices;
	ifstream inFile;

	inFile.open("um/all.idx");

	unsigned long long count = 0;

	cout << "Reading data from all.idx" << endl;

	while (!inFile.eof()) {

		int x;

		inFile >> x;

		if (x == DATASET_NUM) {

			list_of_indices.push_back(count);

		}

		if (count % 1000000 == 0) {

			cout << "\r" << to_string(count * 100 / 102416306) << "%%" << flush;

		}

		count++;

	}

	inFile.close();

	cout << endl;



	// shuffle

	cout << "Shuffling indices" << endl;

	srand(unsigned(time(0)));
	cout << list_of_indices.size()<<endl;
	random_shuffle(list_of_indices.begin(), list_of_indices.end());


	// read data from all.dta and print the once in list_of_indices

	cout << "Selecting datapoints from all.dta" << endl;

	inFile.open("um/all.dta");

	int* user_matrix = new int[READ_IN_LINES];
	short* movie_matrix = new short[READ_IN_LINES];
	char* rating_matrix = new char[READ_IN_LINES];
	short*  time_matrix = new short[READ_IN_LINES];
	for (int i = 0; i < TRAIN_SIZE; i++) {
		inFile >> user_matrix[i];
		inFile >> movie_matrix[i];
		inFile >> time_matrix[i];
		inFile >> rating_matrix[i];
	}

	ofstream outFile;

	outFile.open("dataset" + to_string(DATASET_NUM) + "_random_samples_" + to_string(TRAIN_SIZE) + ".dta");

	long line_num = 0;

	long p = 0;

	long user, movie, date, rating;

	while (p < list_of_indices.size()) {
		outFile << user_matrix[list_of_indices[p]] << " " << movie_matrix[list_of_indices[p]] << " " << time_matrix[list_of_indices[p]] << " " << rating_matrix[list_of_indices[p]] << endl;
			if (p % 1000 == 0) {
				cout << "\r" << to_string(p / TRAIN_SIZE) << "%%" << flush;
			}

			p++;

	}

	inFile.close();

	outFile.close();

	cout << p << endl;



	return 0;

}