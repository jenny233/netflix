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
#define DO_SHUFFLE true  // true: time seeded shuffle, false: no shuffling

#define DATASET_NUM  2  // The dataset that we want to select
#define OUTFILE_NAME "dataset2_shuffled_all.dta"



using namespace std;



int main () {

	// read in the indices
	vector <int> list_of_indices;
	ifstream inFile;
	inFile.open("../mu/all.idx");

	unsigned long long count = 0;
	cout << "Reading data from all.idx" << endl;

	while (count < READ_IN_LINES) {

		int x;
		inFile >> x;

		if (x == DATASET_NUM) {
			list_of_indices.push_back(count);
		}

		if (count % 1000000 == 0) {
			cout << "\r" << to_string(count * 100 / READ_IN_LINES) << "%%" << flush;
		}

		count++;
	}

	inFile.close();
	cout << endl;
	cout << "Number of lines extracted: " << list_of_indices.size()<<endl;



	// shuffle
	if (DO_SHUFFLE) {
		cout << "Shuffling indices" << endl;
		srand ( unsigned ( time(0) ) );
	    random_shuffle ( list_of_indices.begin(), list_of_indices.end() );
	}



	// read data from all.dta and print the once in list_of_indices
	cout << "Selecting datapoints from all.dta" << endl;

	inFile.open("../mu/all.dta");

	int* user_matrix = new int[READ_IN_LINES];
	short* movie_matrix = new short[READ_IN_LINES];
	short* rating_matrix = new short[READ_IN_LINES];
	short*  time_matrix = new short[READ_IN_LINES];
	for (long i = 0; i < READ_IN_LINES; i++) {
		inFile >> user_matrix[i];
		inFile >> movie_matrix[i];
		inFile >> time_matrix[i];
		inFile >> rating_matrix[i];
		if (i % 1000000 == 0) {
			cout << "\r" << to_string(i * 100 / READ_IN_LINES) << "%%" << flush;
		}
	}
	inFile.close();
	cout << endl;



	// write extracted data to output file
	ofstream outFile;
	outFile.open(OUTFILE_NAME);

	for (long p = 0; p < list_of_indices.size(); p++) {

		outFile << user_matrix[list_of_indices[p]] << " "
		<< movie_matrix[list_of_indices[p]] << " "
		<< time_matrix[list_of_indices[p]] << " "
		<< rating_matrix[list_of_indices[p]] << endl;

		if (user_matrix[list_of_indices[p]] <= 0) {
			cout << "THIS LINE IS WRONG!!!" << endl;
			cout << "p: " << p << " line: " << list_of_indices[p] << " user: " << user_matrix[list_of_indices[p]] << endl;
		}
		if (p % 1000000 == 0) {
			cout << "\r" << to_string(p * 100 / READ_IN_LINES) << "%%" << flush;
		}
	}

	outFile.close();



	return 0;

}
