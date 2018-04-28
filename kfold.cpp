#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include "SVD.h"

using namespace std;

function kfold_train (list of indices or arrays)
{
	/*
	* This function does SVD on the given indices and then validates 
	* Inputs: 
	* Outputs:
	*/

	// train on the indices using SVD

	// predict ratings of test points

	// compute error of test points

	// return error and predictions (with corresponding indices)
}

function kfold (array users, array movies, array ratings, array times, int splits)
{
	/*
	* This function performs k-fold cross validation on arrays of data
	*/
	int start;
	int end;
	float val_errors[splits];

	// size of each split
	int split_size = data_size / splits;
	for (int i = 0; i < splits; i++)
	{
		// first create the arrays to give to SVD
		// set the start point for testing
		start = split_size * i;
		// set the end point for testing
		end = split_size * (i + 1);
		// unless we're at the last split, so we want to use all the remaining points
		if (i == splits - 1)
		{
			end = split_size;
		}

		review_size = users.size();
		
		// set the size of the train and test arrays
		int train_users[review_size - end + start];
		short train_movies[review_size - end + start];
		short train_times[review_size - end + start];
		char train_ratings[review_size - end + start];

		int test_users[end - start];
		short test_movies[end - start];
		short test_times[end - start];
		char test_ratings[end - start];

		for (int j = 0; j < users.size(); j++)
		{
			// point is in training array if it's outside bounds 
			// otherwise it's in the test array
			if (j < start)
			{
				train_users[j] = users[j];
				train_movies[j] = movies[j];
				train_times[j] = times[j];
				train_ratings[j] = ratings[j];
			}
			else if (j >= start && j < end)
			{
				test_users[start - j] = users[j];
				test_movies[start - j] = movies[j];
				test_times[start - j] = times[j];
				test_ratings[start - j] = ratings[j];
			}
			else
			{
				train_users[j - end + start] = users[j];
				train_movies[j] = movies[j];
				train_times[j] = times[j];
				train_ratings[j] = ratings[j];
			}
		}

		// perform SVD with validation

		float val_error = train_model(int M, int N, int K, float eta, float reg,
			train_users[], train_movies[], train_times[], train_ratings[],
			test_users[], test_movies[], test_times[], test_ratings[],
			float eps = 0.001, int max_epochs = EPOCH);

		// validation error is returned by the SVD

		// store the validation error in an array for each fold 

		val_errors[i] = val_error;

		cout << "The validation error on fold " << i << " is " << val_error << endl;
	}

	float sum;
	for (int k = 0; k < val_errors.size(), k++)
	{
		sum += val_errors[k];
	}

	return k / val_errors.size();
}