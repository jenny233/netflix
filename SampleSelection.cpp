#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>    // std::shuffle

#define TRAIN_SIZE  500000
#define DATASET_NUM  1  // The dataset that we want to select

using namespace std;

int main() {
    vector <int> list_of_indices;

    ifstream inFile;
    inFile.open("../mu/all.idx");
    long count = 0;
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
        count ++;
    }
    inFile.close();
    cout << endl;

    // shuffle
    cout << "Shuffling indices" << endl;
    srand ( unsigned ( time(0) ) );
    random_shuffle ( list_of_indices.begin(), list_of_indices.end() );

    // sort
    cout << "Sorting first " << TRAIN_SIZE << " indices" << endl;
    sort (list_of_indices.begin(), list_of_indices.begin() + TRAIN_SIZE);

    // read data from all.dta and print the once in list_of_indices
    cout << "Selecting datapoints from all.dta" << endl;
    inFile.open("../mu/all.dta");
    ofstream outFile;
    outFile.open("dataset"+to_string(DATASET_NUM)+"_random_samples_"+to_string(TRAIN_SIZE)+".dta");

    long line_num = 0;
    long p = 0;
    long user, movie, date, rating;
    while (p < TRAIN_SIZE) {
        inFile >> user >> movie >> date >> rating;
        if (line_num == list_of_indices[p]) {
            outFile << user << " " << movie << " " << date << " " << rating << endl;
            if (p % 1000 == 0) {
                cout << "\r" << to_string(p / 5000) << "%%" << flush;
            }
            p ++;
        }
        line_num ++;
    }
    inFile.close();
    outFile.close();
    cout << endl;

    return 0;
}
