#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include "config.hpp"

vector<vector<Point> > deleting_elements(vector<double> &SW_result, vector<double> &PD_result, vector<double> &eHOG_result, vector<vector<Point> > regions);
Mat equalized (Mat img);
void scheletro(Mat source, Mat *skel);
Mat hog_original (Mat source);


#endif // _UTILS_HPP_
