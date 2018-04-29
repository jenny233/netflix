#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <string>
#define USER_SIZE 458294
#define READ_IN_LINES 94362235
#define MOVIE_SIZE 17771

using namespace std;

vector<int>* adj_user = new vector<int> [USER_SIZE];
vector<int>* adj_movie = new vector<int> [MOVIE_SIZE];
int num_of_unique_movie;
int num_of_unique_user;

void array_pop_u( int user_matrix[], char  rating_matrix[], char pop)
{
	long long counter = 0;
	for (long long i = 0; i < READ_IN_LINES ; i++)
	{
		char rating = rating_matrix[i];
		{
			(adj_user[user_matrix[i]]).push_back(rating - '0');
		}
		counter++;
	}

	if (pop == 'u')
		num_of_unique_user = counter;
	else
		num_of_unique_movie = counter;
}
void array_pop_m(short user_matrix[], char rating_matrix[], char pop)
{
	int counter = 0;
	for (int i = 0; i < READ_IN_LINES; i++)
	{
		(adj_movie[user_matrix[i]]).push_back(rating_matrix[i] - '0');
		counter++;
	}

	if (pop == 'u')
		num_of_unique_user = counter;
	else
		num_of_unique_movie = counter;
}


double baseline_pred(int user, int movie)
{
	/* Finds more simple predictor of movie baselines */
	double user_average = 0.0;
	double movie_average = 0.0;

	for (unsigned int i = 0; i < (adj_user[user]).size(); i++)
		user_average += (double)(adj_user[user])[i];
	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		movie_average += (double)(adj_movie[movie])[i];
	return ((user_average / (double)((adj_user[user]).size())) + (movie_average / (double)(adj_movie[movie]).size())) / 2.0;
}

double baseline_pred_better(int user, int movie, double avg)
{
	double total = 0;
	double b_i, b_u;
	double lamdba1 = 25.0;
	double lamdba2 = 10.0;
	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		total += (double)(adj_movie[movie])[i] - avg;
	b_i = total / (lamdba1 + ((adj_movie[movie]).size()));
	total = 0;
	for (unsigned int i = 0; i < (adj_user[user].size()); i++)
		total += (double)(adj_user[user])[i] - avg - b_i;
	b_u = total / (lamdba2 + ((adj_user[user]).size()));
	return (b_u + b_i + avg);
}

vector<double> find_bias(int user, int movie, double avg)
{
	double total = 0;
	double b_i, b_u;
	double lamdba1 = 25.0;
	double lamdba2 = 10.0;
	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		total += (double)(adj_movie[movie])[i] - avg;
	b_i = total / (lamdba1 + ((adj_movie[movie]).size()));
	double total2 = 0;
	for (unsigned int i = 0; i < (adj_user[user]).size(); i++)
		total2 += (double)((adj_user[user])[i]) - avg - b_i;
	b_u = total2 / (lamdba2 + ((adj_user[user]).size()));
	vector <double> bias;
	bias.push_back(b_i);
	bias.push_back(b_u);
	return bias;

}


double SGD_bias_u(double u_bias[], double m_bias[], int user_matrix[], short movie_matrix[], char rating_matrix[], double avg, int rating_index, double learning_rate, double lamdba)
{
	double gradient = 0.0;
	gradient = -2.0 * ((rating_matrix[rating_index] - '0') - avg - u_bias[user_matrix[rating_index]] - m_bias[movie_matrix[rating_index]]) + lamdba * 2.0 * u_bias[user_matrix[rating_index]];
	u_bias[user_matrix[rating_index]] -= learning_rate * gradient;
	return u_bias[user_matrix[rating_index]];
}

double SGD_bias_m(double u_bias[], double m_bias[], int user_matrix[], short movie_matrix[], char rating_matrix[], double avg, int rating_index, double learning_rate, double lamdba)
{
	double gradient = 0.0;
	gradient = -2.0 * ((rating_matrix[rating_index] - '0') - avg - u_bias[user_matrix[rating_index]] - m_bias[movie_matrix[rating_index]]) + lamdba * 2.0 * m_bias[movie_matrix[rating_index]];
	m_bias[movie_matrix[rating_index]] -= learning_rate * gradient;
	return m_bias[movie_matrix[rating_index]];
}

double calc_error(double u_bias[], double m_bias[], int user_matrix[], short movie_matrix[], char rating_matrix[], double avg)
{
	double error = 0.0;
	for (int j = 0; j < READ_IN_LINES; j++)
	{
		error += pow(((double) ( rating_matrix[j] - '0') - avg - u_bias[user_matrix[j]] - m_bias[movie_matrix[j]]), 2.0);
	}
	return (error / READ_IN_LINES);
}

void checkpoint(double u_bias[], double m_bias[], double avg)
{
	ofstream outputFile;
	outputFile.open("bias_checkpoint.txt");
	outputFile << "average:" << endl;
	outputFile << avg << endl;
	outputFile << "user_bias:" << endl;
	for (long i = 0; i < USER_SIZE; i++)
	{
		outputFile << u_bias[i] << endl;
	}
	outputFile << "movie_bias:" << endl << endl;
	for (long i = 0; i < MOVIE_SIZE; i++)
	{ 
		outputFile << m_bias[i] << endl;
	}
	outputFile.close();
}

void gradient_descent_bias(double u_bias[], double m_bias[], int user_matrix[], short movie_matrix[], char rating_matrix[], double lamdba, double avg)
{
	double error = 0.000;
	double learning_rate = 0.1;
	double new_error = calc_error(u_bias, m_bias, user_matrix, movie_matrix, rating_matrix, avg);
	int counter = 0;
	while (abs(error - new_error) / error > (.000001))
	{
		if (counter > 1000)
			learning_rate = .01;
		error = new_error;
		for (int j = 0; j < READ_IN_LINES; j++)

		{
			u_bias[user_matrix[j]] = SGD_bias_u(u_bias, m_bias, user_matrix,movie_matrix,rating_matrix, avg, j, learning_rate,lamdba);
			m_bias[movie_matrix[j]] = SGD_bias_m(u_bias, m_bias, user_matrix, movie_matrix, rating_matrix, avg, j, learning_rate, lamdba);
		}
		
		cout << "Iteration " << counter << endl;
		new_error = calc_error(u_bias, m_bias, user_matrix, movie_matrix, rating_matrix, avg);
		counter += 1;
	}
	cout << counter << endl;
	cout << new_error << endl;
	checkpoint(u_bias, m_bias, avg);
}

int main()
{

	for (long i = 0; i < USER_SIZE; i++)
	{
		vector <int> m;
		adj_user[i] = m;
	}

	for (long i = 0; i < MOVIE_SIZE; i++)
	{
		vector <int> m;
		adj_movie[i] = m;
	}

	ifstream inFile;
	inFile.open("dataset1_random_samples_all.dta");
	if (!inFile) {
		std::cout << "File not opened." << endl;
		exit(1);

	}
	// The matrix of all of the ratings
	int* user_matrix = new int [READ_IN_LINES];
	short* movie_matrix = new short[READ_IN_LINES];
	char* rating_matrix = new char[READ_IN_LINES];
	double total_average_movie_rating = 0;
	int temp;
	for (long long i = 0; i < READ_IN_LINES; i++) {
			inFile >> user_matrix[i];
			inFile >> movie_matrix[i];
			inFile >> temp;
			inFile >> rating_matrix[i];
		total_average_movie_rating += rating_matrix[i] - '0';
		if (i % 1000000 == 0)
			cout << "\r" << to_string(i * 100 / 102416306) << "%%" << flush;
	}

	total_average_movie_rating = total_average_movie_rating / READ_IN_LINES;
	cout << "pass" << endl;
	array_pop_m(movie_matrix, rating_matrix, 'm');
	array_pop_u(user_matrix, rating_matrix, 'u');


	double error1 = 0;
	double error2 = 0;
	double* bias_user = new double[USER_SIZE];
	double* bias_movie = new double[MOVIE_SIZE];
	for (int i = 0; i < USER_SIZE; i++)
	{
		bias_user[i] = 0.0;
	}

	for (int i = 0; i < MOVIE_SIZE; i++)
	{
		bias_movie[i] = 0.0;
	}

	gradient_descent_bias(bias_user, bias_movie, user_matrix,movie_matrix,rating_matrix, 0.1, total_average_movie_rating);

	cin.get();

	delete[] rating_matrix;
	delete[] user_matrix;
	delete[] movie_matrix;
	delete[] bias_movie;
	delete[] bias_user;
	delete[] adj_user;
	return 0;


}