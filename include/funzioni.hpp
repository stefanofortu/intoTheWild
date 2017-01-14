/*
 * HEADER FILE
 */

#ifndef FUNZIONI_HPP
#define FUNZIONI_HPP

#include "config.hpp"


Mat Bayes(double positive_dist_SW[51], double negative_dist_SW[51], double positive_dist_PD[51],
                    double negative_dist_PD[51], double positive_dist_EHOG[51], double negative_dist_EHOG[51],
                    vector<double> SW, vector<double> PD, vector<double> EHOG, Mat source, vector<vector<Point> > regions);

Mat hog_original (Mat source);

vector<double> ComputeEHOG(Mat source,  vector<vector<Point> > regions);
vector<double> ComputePD(Mat source,  vector<vector<Point> > regions);
vector<double> ComputeStrokeWidth(Mat source,  vector<vector<Point> > regions);
vector<int> regions_cue(vector<double> cue, double positive_dist[51], double negative_dist[51], Mat source, vector<vector<Point> > regions, int numero, Mat *risultato);


#endif // FUNZIONI_HPP
