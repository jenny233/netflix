#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <queue>
#include <cmath>

#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
using namespace std;

#define NUM_USERS 458293
//#define NUM_MOVIES 17770
#define NUM_MOVIES 17770
#define MAX_CHARS_PER_LINE 30
#define MIN_COMMON 16

struct mu_pair
{
	unsigned int user;
	unsigned char rating;
};

struct um_pair
{
	unsigned short movie;
	unsigned char rating;
};

// To be stored in P
struct s_pear {
	float p;
	unsigned int common;
};


// Used during prediction
// As per the blogpost
struct s_neighbors {
	// Num users who watched both m and n
	unsigned int common;

	// Avg rating of m, n
	float m_avg;
	float n_avg;

	// Rating of n
	float n_rating;

	// Pearson coeff
	float pearson;

	float p_lower;
	float weight;
};

// Comparison operator for s_neighors
int operator<(const s_neighbors &a, const s_neighbors &b) {
	return a.weight > b.weight;
}

// Pearson intermediates
struct s_inter
{
	float x; // sum of ratings of movie i
	float y; // sum of ratings of movie j
	float xy; // sum (rating_i * rating_j)
	float xx; // sum (rating_i^2)
	float yy; // sum (rating_j^2)
	unsigned int n; // Num users who rated both movies
};

class KNN {
private:
	// um: for every user, stores (movie, rating) pairs.
	vector<um_pair> um[NUM_USERS];

	// mu: for every movie, stores (user, rating) pairs.
	vector<mu_pair> mu[NUM_MOVIES];


	// Pearson coefficients for every movie pair
	// When accessing P[i][j], it must always be the case that:
	// i <= j (symmetry is assumed)
	s_pear P[NUM_MOVIES][NUM_MOVIES];

	double predictRating(unsigned int movie, unsigned int user);
	//void outputRMSE(short numFeats);
	stringstream mdata;

	float movieAvg[NUM_MOVIES]; // average movie rating
public:
	KNN();
	~KNN() { };
	void loadData();
	void calcP();
};

KNN::KNN()
{
	mdata << "-KNN-";
}

void KNN::loadData()
{
	// um, mu - userId, movieId, rating
	// movieAvg array - average rating for each movie
}

void KNN::calcP() {
	int i, u, m, user, z;
	double rmse, rmse_last;
	short movie;
	float x, y, xy, xx, yy;
	unsigned int n;

	char rating_i, rating_j;

	// Vector size
	int size1, size2;

	// Intermediates for every movie pair
	s_inter tmp[NUM_MOVIES];

	cout << "Calculating P" << endl;

	rmse_last = 0;
	rmse = 2.0;

	float tmp_f;


	// Compute intermediates
	for (i = 0; i < NUM_MOVIES; i++) {

		// Zero out intermediates
		for (z = 0; z < NUM_MOVIES; z++) {
			tmp[z].x = 0;
			tmp[z].y = 0;
			tmp[z].xy = 0;
			tmp[z].xx = 0;
			tmp[z].yy = 0;
			tmp[z].n = 0;
		}

		size1 = mu[i].size();

		if ((i % 100) == 0) {
			cout << i << endl;
		}

		// For each user that rated movie i
		for (u = 0; u < size1; u++) {
			user = mu[i][u].user;

			size2 = um[user].size();
			// For each movie j rated by current user
			for (m = 0; m < size2; m++) {
				// id of movie j
				movie = um[user][m].movie; 

				// At this point, we know that user rated both movie i AND movie j
				// Thus we can update the pearson coeff for the pair XY

				// Rating of movie i
				rating_i = mu[i][u].rating;

				// Rating of movie j
				rating_j = um[user][m].rating;

				// Increment rating of movie i
				tmp[movie].x += rating_i;

				// Increment rating of movie j
				tmp[movie].y += rating_j;

				tmp[movie].xy += rating_i * rating_j;
				tmp[movie].xx += rating_i * rating_i;
				tmp[movie].yy += rating_j * rating_j;

				// Increment number of viewers of movies i AND j
				tmp[movie].n += 1;
			}
		}

		// Calculate Pearson coeff. based on: 
		// https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
		for (z = 0; z < NUM_MOVIES; z++) {
			x = tmp[z].x;
			y = tmp[z].y;
			xy = tmp[z].xy;
			xx = tmp[z].xx;
			yy = tmp[z].yy;
			n = tmp[z].n;
			if (n == 0) {
				P[i][z].p = 0;
			}
			else {
				// tmp_f = (n * xy - x * y) / (sqrt((n- 1) * xx - x*x) * sqrt((n - 1) * yy - (y * y)));
				tmp_f = (n * xy - x * y) / (sqrt(n * xx - x * x) * sqrt(n * yy - y * y));
				// Test for NaN
				if (tmp_f != tmp_f) {
					tmp_f = 0.0;
				}
				P[i][z].p = tmp_f;
				P[i][z].common = n;
			}
		}
		//cout << P[i][z].p << endl;
	}
	cout << "P calculated" << endl;
	
}

double KNN::predictRating(unsigned int movie, unsigned int user) {

	double prediction = 0;
	double denom = 0;
	double diff;
	double result;

	unsigned int size, i, n;

	s_pear tmp;
	s_neighbors neighbors[NUM_MOVIES];
	priority_queue<s_neighbors> q;
	s_neighbors tmp_pair;
	float p_lower, pearson;
	int common_users;

	// Len neighbors
	int j = 0;

	// For each movie rated by user
	size = um[user].size();

	for (i = 0; i < size; i++) {
		n = um[user][i].movie; // n: movie watched by user

		tmp = P[min(movie, n)][max(movie, n)];
		common_users = tmp.common;

		// If movie and m2 have >= MIN_COMMON viewers
		if (common_users >= MIN_COMMON) {
			neighbors[j].common = common_users;
			neighbors[j].m_avg = movieAvg[movie];
			neighbors[j].n_avg = movieAvg[n];

			neighbors[j].n_rating = um[user][i].rating;

			pearson = tmp.p;
			neighbors[j].pearson = pearson;

			// Fisher and inverse-fisher transform (from wikipedia)
			p_lower = tanh(atanh(pearson) - 1.96 / sqrt(common_users - 3));
			//p_lower = pearson;
			neighbors[j].p_lower = p_lower;
			neighbors[j].weight = p_lower * p_lower * log(common_users);
			j++;
		}

	}

	// Add the dummy element described in the blog
	neighbors[j].common = 0;
	neighbors[j].m_avg = movieAvg[movie];
	neighbors[j].n_avg = 0;

	neighbors[j].n_rating = 0;

	neighbors[j].pearson = 0;

	neighbors[j].p_lower = 0;
	neighbors[j].weight = log(MIN_COMMON);
	j++;
	
	result = ((float)prediction) / denom;

	return result;
}


int main()
{
	KNN *knn = new KNN();
	knn->loadData();
	knn->calcP();
	cin.get();
	return 0;
}
