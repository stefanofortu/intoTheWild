#include "utils.hpp"
#include "histogram.hpp"

vector<vector<Point> > deleting_elements(vector<double> &SW_result, vector<double> &PD_result, vector<double> &eHOG_result, vector<vector<Point> > regions);
Mat equalized (Mat img);
void scheletro(Mat source, Mat *skel);
Mat hog_original (Mat source);



/*
 * This method is used to delete regions that don't satisfy the constraint related to the skeleton size;
 * is checked the content of SW_result[i] and if it's < 0 (imposed in the method "computeStrokeWidth")
 * the region analyzed has be deleted from each result, so from SW_result, PD_result, eHOG_result and
 * obviously also from regions.
 */
vector<vector<Point> > deleting_elements(vector<double> &SW_result, vector<double> &PD_result, vector<double> &eHOG_result, vector<vector<Point> > regions)
{
    cout << "ATTENTION! THE INITIAL SIZE IS: " << SW_result.size() << endl;
    for(unsigned int i=0; i<SW_result.size();i++)
    {
        if(SW_result[i]<0)
        {
            regions.erase(regions.begin()+i);
            SW_result.erase(SW_result.begin()+i);
            PD_result.erase(PD_result.begin()+i);
            eHOG_result.erase(eHOG_result.begin()+i);
            i--;
        }
    }
    cout << "HEY! NOW THE SIZE IS: " << SW_result.size() << endl;
    return regions;
}

Mat equalized (Mat img)
{
    int i,j,k;
    Mat eq(img.rows,img.cols,CV_8UC1);
    float* v = histogram_gray (img);
    float* z = cumulative (v);
    for(i=0;i<img.rows;i++)
    {
        for(j=0;j<img.cols;j++)
        {
            k = (img).at<unsigned char>(i,j);
            (eq).at<unsigned char>(i,j)=z[k]*255;
        }
    }
    return eq;
}

/*
 * This method is used to compute the skeleton of an image using morphological operator
 */
void scheletro(Mat source, Mat *skel)
{
    //imshow("source", source);

    //This represents the kernel used for the morphological operations
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

    bool done = false;

    Mat eroded = eroded.zeros(source.size(), source.type());
    Mat out_dilate = out_dilate.zeros(source.size(), source.type());

    do
    {
        //Processing of iterative erosion to get the skeleton
        erode(source,eroded, element);
        dilate(eroded, out_dilate, element);
        subtract(source,out_dilate, out_dilate);
        bitwise_or(*skel, out_dilate, *skel);
        eroded.copyTo(source);

        double max;
        minMaxLoc(source, 0, &max, 0, 0);
        done = (max == 0);

    }while(!done);

//    imshow("skel",*skel);
//    waitKey();
}


/*
 * This method is used to compute the gradient of each pixel obtained by computing the square root of the sum of the
 * squared gradients along x and y drections.
 */
Mat hog_original (Mat source)
{
    if(source.type()==CV_8UC3)
    cvtColor(source, source, CV_RGB2GRAY);

    Mat kernel_gradient_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat kernel_gradient_y = (Mat_<float>(3,3) << 1,2,1,0,0,0,-1,-2,-1);

    Mat gradient_x(source.size(), CV_32FC1), gradient_y(source.size(), CV_32FC1), uscita(source.size(), CV_32FC1);
    filter2D(source, gradient_x, CV_32F, kernel_gradient_x);
    filter2D(source, gradient_y, CV_32F, kernel_gradient_y);

    uscita = gradient_x.mul(gradient_x)+gradient_y.mul(gradient_y);

    for(int rows = 0; rows < uscita.rows; rows++)
        for(int cols = 0; cols < uscita.cols; cols++)
            uscita.at<float>(rows,cols)=sqrt(uscita.at<float>(rows,cols));

    normalize(uscita,uscita, 0, 255, NORM_MINMAX,CV_32F);
    return uscita;
}



