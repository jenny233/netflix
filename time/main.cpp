#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <string>
#include <string.h>
#include <utility>
#include "SVD.cpp"

using namespace std;

int main() {
    string trainFile = "../dataset1_shuffled_all.dta";  //set train data
    string crossFile = "../dataset2_shuffled_all.dta";  //set cross validation data
    string testFile = "../dataset5_unshuffled_all.dta";  //set test data
    string outFile = "5120309085_5120309016_5120309005.txt";  //set output data
    ifstream fp("training.txt");
    ofstream ft("train.txt");
    ofstream fc("cross.txt");
    srand(time(NULL));
    char s[2048];
    while (fp.getline(s, 2000)) {
        if (rand()%100==0) {
            fc << s << endl;
        
        }
        else {
            ft << s << endl;
        }
    }
    cout<<"hi"<<endl;
    fp.close();
    ft.close();
    fc.close();
    SVD svd(NULL,NULL,0,NULL,NULL, trainFile, crossFile, testFile, outFile);
    double rmse = svd.MyTrain();     //train
    return 0;
}
