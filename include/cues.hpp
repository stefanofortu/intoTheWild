#ifndef _CUES_HPP_
#define _CUES_HPP_

#include "config.hpp"

vector<double> ComputeEHOG(Mat source,  vector<vector<Point> > regions);
vector<double> ComputePD(Mat source,  vector<vector<Point> > regions);
vector<double> ComputeStrokeWidth(Mat source,  vector<vector<Point> > regions);
vector<int> regions_cue(vector<double> cue, double positive_dist[51], double negative_dist[51], Mat source, vector<vector<Point> > regions, int numero, Mat *risultato);


Mat Bayes(double positive_dist_SW[51], double negative_dist_SW[51], double positive_dist_PD[51],
                    double negative_dist_PD[51], double positive_dist_EHOG[51], double negative_dist_EHOG[51],
                    vector<double> SW, vector<double> PD, vector<double> EHOG, Mat source, vector<vector<Point> > regions);



#endif // _CUES_HPP_
