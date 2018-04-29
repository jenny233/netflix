#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>    // std::shuffle

// #define SELECTION_SIZE
#define DATASET_NUM  2  // The dataset that we want to select

using namespace std;

int main() {
    vector <int> list_of_indices;

    ifstream inFile;
    inFile.open("../mu/all.idx");
    long total_lines = 0;
    long needed_lines = 0;
    cout << "Reading data from all.idx" << endl;
    while (!inFile.eof()) {
        int x;
        inFile >> x;
        if (x == DATASET_NUM) {
            list_of_indices.push_back(total_lines);
            needed_lines ++;
        }
        if (total_lines % 1000000 == 0) {
            cout << "\r" << to_string(total_lines * 100 / 102416306) << "%%" << flush;
        }
        total_lines ++;
    }
    inFile.close();
    cout << endl;
    cout << needed_lines << " lines read from dataset " << DATASET_NUM << endl;

    // shuffle
    // cout << "Shuffling indices" << endl;
    // srand ( unsigned ( time(0) ) );
    // random_shuffle ( list_of_indices.begin(), list_of_indices.end() );

    // sort
    // cout << "Sorting first " << TRAIN_SIZE << " indices" << endl;
    // sort (list_of_indices.begin(), list_of_indices.begin() + TRAIN_SIZE);

    // read data from all.dta and print the once in list_of_indices
    cout << "Selecting datapoints from all.dta" << endl;
    inFile.open("../mu/all.dta");
    ofstream outFile;
    outFile.open("dataset"+to_string(DATASET_NUM)+"_random_samples_all.dta");

    long line_num = 0;
    long p = 0;
    long user, movie, date, rating;
    while (!inFile.eof() && p < needed_lines) {
        inFile >> user >> movie >> date >> rating;
        if (line_num == list_of_indices[p]) {
            outFile << user << " " << movie << " " << date << " " << rating << endl;
            if (p % 100000 == 0) {
                cout << "\r" << to_string(p * 100 / needed_lines) << "%%" << flush;
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
