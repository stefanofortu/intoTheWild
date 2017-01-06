/*
 * HEADER FILE
 */

#ifndef FUNZIONI_HPP
#define FUNZIONI_HPP
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "../vlfeat/vl/mser.h"

using namespace std;
using namespace cv;

#define _DIFFERENCE  0
#define _MSE         1
#define _PSNR        2
#define POSITIVE_SAMPLES 4781
#define NEGATIVE_SAMPLES 5100
#define MSER_BRIGHT_ON_DARK 0
#define MSER_DARK_ON_BRIGHT 1
#define prioriN 0.3
#define prioriT 0.7
/*
 * FUNCTIONS PROTOTYPES
 */
float* cumulative(float* histogram);
float* histogram_gray (Mat img);
int caricaDistribuzione(string address, double* distribution);
int histc(double *distribution_source, int n_samples, double step, double *hist_distribution);
int PreEMSER(Mat source,Mat* brightGrayImg_OUT, Mat*darkGrayImg_OUT, bool show);
int vl_feat_mser(Mat input_img, vector <vector <Point > > & regioni_FF , Mat* output_img ,int mser_type);
Mat Bayes(double positive_dist_SW[51], double negative_dist_SW[51], double positive_dist_PD[51],
                    double negative_dist_PD[51], double positive_dist_EHOG[51], double negative_dist_EHOG[51],
                    vector<double> SW, vector<double> PD, vector<double> EHOG, Mat source, vector<vector<Point> > regions);
Mat equalized (Mat img);
Mat hog_original (Mat source);
Mat plot_histogram(float* istogramma);
vector<double> ComputeEHOG(Mat source,  vector<vector<Point> > regions);
vector<double> ComputePD(Mat source,  vector<vector<Point> > regions);
vector<double> ComputeStrokeWidth(Mat source,  vector<vector<Point> > regions);
vector<int> regions_cue(vector<double> cue, double positive_dist[51], double negative_dist[51], Mat source, vector<vector<Point> > regions, int numero, Mat *risultato);
vector<vector<Point> > deleting_elements(vector<double> &SW_result, vector<double> &PD_result, vector<double> &eHOG_result, vector<vector<Point> > regions);
void scheletro(Mat source, Mat *skel);
#endif // FUNZIONI_HPP
