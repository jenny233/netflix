#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#define TRAIN_SIZE  500000
#define N_ENTRIES  4
#define USER_SIZE  500000
#define MOVIE_SIZE  5000
#define EPOCH  500

using namespace std;

int Y_train [TRAIN_SIZE][N_ENTRIES];

int main() {

    ifstream inFile;
    inFile.open("../mu/all.dta");

    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }

    for (int i=0; i<TRAIN_SIZE; i++) {
        for (int j=0; j<N_ENTRIES; j++) {
            inFile >> Y_train[i][j];
        }
    }





    return 0;
}
