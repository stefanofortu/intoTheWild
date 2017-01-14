#ifndef _MSER_WARP_HPP_
#define _MSER_WARP_HPP_

#include "config.hpp"

int PreEMSER(Mat source,Mat* brightGrayImg_OUT, Mat*darkGrayImg_OUT, bool show);
int vl_feat_mser(Mat input_img, vector <vector <Point > > & regioni_FF , Mat* output_img ,int mser_type);

#endif //_MSER_WARP_HPP