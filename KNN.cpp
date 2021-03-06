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
#include <chrono>  

#define NUM_USERS 458294
#define NUM_MOVIES 17771
#define NUM_RATINGS 98291669
#define GLOBAL_AVG 3.512599976023349
#define GLOBAL_OFF_AVG 0.0481786328365
#define NUM_PROBE_RATINGS 1374739
#define MAX_CHARS_PER_LINE 30
#define TRAIN_SIZE 94362233

// Minimum common neighbors required for decent prediction
#define MIN_COMMON 16
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

// Max weight elements to consider when predicting
#define MAX_W 10

// Ideas and some pseudocode from: http://dmnewbie.blogspot.com/2009/06/calculating-316-million-movie.html


using namespace std;

struct mu_pair {
	unsigned int user;
	unsigned char rating;
};

struct um_pair {
	unsigned short movie;
	unsigned char rating;
};

// Pearson intermediates, as described in dmnewbie's blog
struct s_inter {
	float x; // sum of ratings of movie i
	float y; // sum of ratings of movie j
	float xy; // sum (rating_i * rating_j)
	float xx; // sum (rating_i^2)
	float yy; // sum (rating_j^2)
	unsigned int n; // Num users who rated both movies
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



class KNN {
private:
	// um: for every user, stores (movie, rating) pairs.
	//vector<um_pair> um[NUM_USERS];

	// mu: for every movie, stores (user, rating) pairs.
	//vector<mu_pair> mu[NUM_MOVIES];

	//vector<int>* movies_rated_by_user = new vector<int>[NUM_MOVIES];
	vector<um_pair> movies_rated_by_user[NUM_USERS];
	//vector<int>* user_rated_movies = new vector<int>[NUM_USERS];
	vector<mu_pair> user_rated_movies[NUM_MOVIES];


	// Pearson coefficients for every movie pair
	// When accessing P[i][j], it must always be the case that:
	// i <= j (symmetry is assumed)
	//s_pear P[NUM_MOVIES][NUM_MOVIES];
	s_pear (*P)[NUM_MOVIES] = (s_pear (*)[NUM_MOVIES]) calloc(NUM_MOVIES, NUM_MOVIES);

	double predictRating(unsigned int movie, unsigned int user);
	void outputRMSE(short numFeats);
	stringstream mdata;

	float movieAvg[NUM_MOVIES];
public:
	KNN();
	~KNN() { };
	void loadData();
	void calcP();
	void saveP();
	//void loadP();
	//void output();
	//void save();
	//void probe();
};

KNN::KNN()
{
	mdata << "-KNN-";
}

void KNN::loadData() {
	string line;
	char c_line[MAX_CHARS_PER_LINE];
	int userId;
	int movieId;
	int time;
	int rating;

	int j = -1;

	int i = -1;
	int last_seen = 0;
	int last_seen2 = 0;

	// Used for movie avgs
	int num_ratings = 0;
	int avg = 0;
	
	// populate user array with empty vectors 
	/*
	for (long i = 0; i < NUM_USERS; i++)
	{
		vector <int> m;
		movies_rated_by_user[i] = m;
		user_rated_movies[i] = m;
	}*/

	ifstream inFile;
	inFile.open("dataset1_unshuffled_all2.dta");
	if (!inFile) {
		std::cout << "File not opened." << endl;
		exit(-1);
	}
	for (long k = 0; k < TRAIN_SIZE; k++) {
		inFile >> userId;
		inFile >> movieId;
		inFile >> time;
		inFile >> rating;
		if (k % 100 == 0) {
			cout << "\r" << to_string(k * 100 / TRAIN_SIZE) << "%%" << flush;
		}
		if (last_seen == userId) {
			i++;
		}
		else {
			i = 0;
			last_seen = userId;
		}
		if (last_seen2 == movieId) {
			j++;
		}
		else {
			j = 0;
			last_seen2 = movieId;
		}
		//um[userId].push_back(um_pair());
		//um[userId][i].movie = movieId;
		//um[userId][i].rating = rating;

		int movie = movieId;
		int user = userId;
		(movies_rated_by_user[userId]).push_back(um_pair());
		movies_rated_by_user[userId][i].movie = movieId;
		movies_rated_by_user[userId][i].rating = rating;
		(user_rated_movies[movieId]).push_back(mu_pair());
		user_rated_movies[movieId][j].user = userId;
		user_rated_movies[movieId][j].rating = rating;
	}
	inFile.close();

	cout << "Loaded um" << endl;

}


void KNN::calcP() {
	int i, j, u, m, user, z;
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

	//rmse_last = 0;
	//rmse = 2.0;

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

		size1 = user_rated_movies[i].size();

		if ((i % 100) == 0) {
			cout << i << endl;
		}

		// For each user that rated movie i
		for (u = 0; u < size1; u++) {
			user = user_rated_movies[i][u].user;

			size2 = movies_rated_by_user[user].size();
			// For each movie j rated by current user
			for (m = 0; m < size2; m++) {
				movie = movies_rated_by_user[user][m].movie; // id of movie j

				// We know that user rated both movie i AND movie j
				// Now, update the pearson coeff for the pair XY

				// Rating of movie i
				rating_i = user_rated_movies[i][u].rating;

				// Rating of movie j
				rating_j = movies_rated_by_user[user][m].rating;

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
				//tmp_f = (n * xy - x * y) / (sqrt((n- 1) * xx - x*x) * sqrt((n - 1) * yy - (y * y)));
				tmp_f = (n * xy - x * y) / (sqrt(n * xx - x * x) * sqrt(n * yy - y * y));
				// Test for NaN
				if (tmp_f != tmp_f) {
					tmp_f = 0.0;
				}
				P[i][z].p = tmp_f;
				P[i][z].common = n;
			}
		}

	}

	cout << "P calculated" << endl;

}

void KNN::saveP() {
	int i, j;

	cout << "Saving P" << endl;

	ofstream pfile("knn-p", ios::app);
	if (!pfile.is_open()) {
		cout << "Files for P output: Open failed.\n";
		exit(-1);
	}

	for (i = 0; i < NUM_MOVIES; i++) {
		for (j = i; j < NUM_MOVIES; j++) {
			if (P[i][j].common != 0) {
				pfile << i << " " << j << " " << P[i][j].p << " " << P[i][j].common << endl;
			}
		}
	}
	pfile.close();
	cout << "P saved" << endl;
}
/*
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

	// At this point we have an array of neighbors, length j
	// Find the MAX_W elements of the array using 

	// For each movie-pair in neighbors
	for (i = 0; i < j; i++) {
		// If there is place in queue, just push it
		if (q.size() < MAX_W) {
			q.push(neighbors[i]);
		}

		// Else, push it only if this pair has a higher weight than the top
		// (smallest in top-MAX_W).
		// Remove the current top first
		else {
			if (q.top().weight < neighbors[i].weight) {
				q.pop();
				q.push(neighbors[i]);
			}
		}
	}

	// Now we can go ahead and calculate rating
	size = q.size();
	for (i = 0; i < size; i++) {
		tmp_pair = q.top();
		q.pop();
		diff = tmp_pair.n_rating - tmp_pair.n_avg;
		if (tmp_pair.pearson < 0) {
			diff = -diff;
		}
		prediction += tmp_pair.pearson * (tmp_pair.m_avg + diff);
		denom += tmp_pair.pearson; 

	} 

	result = ((float)prediction) / denom;

	// If result is nan, return avg
	if (result != result) {
		return GLOBAL_AVG;
	}
	else if (result < 1) {
		return 1;
	}
	else if (result > 5) {
		return 5;
	}

	return result;

}
*/
int main() {
	KNN *knn = new KNN();
	knn->loadData();
	knn->calcP();
	//knn->saveP();
	//knn->loadP();
	//knn->output();
	//knn->save();
	//knn->probe();
	cout << "KNN completed.\n";
	cin.get();
	return 0;
}
