#ifndef _HISTOGRAM_HPP_
#define _HISTOGRAM_HPP_

#include "config.hpp"
int caricaDistribuzione(string address, double* distribution);
Mat plot_histogram(float* istogramma);

float* cumulative(float* histogram);
float* histogram_gray (Mat img);
int histc(double *distribution_source, int n_samples, double step, double *hist_distribution);

#endif //_HISTOGRAM_HPP_
