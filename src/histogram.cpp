#include "histogram.hpp"

int caricaDistribuzione(string address, double* distribution);
Mat plot_histogram(float* istogramma);

float* cumulative(float* histogram);
float* histogram_gray (Mat img);
int histc(double *distribution_source, int n_samples, double step, double *hist_distribution);

/*
 * This method is used to load distributions of the histograms of the three methods froma a file.
 */
int caricaDistribuzione(string address, double* distribution)
{
    FILE* pFile=fopen(address.c_str(), "r");

    if (pFile !=NULL)
    {
        cout <<"File aperto con successo" << endl;
        rewind(pFile);

       double* ptr=distribution;
       while(!feof(pFile))
        {
            fscanf(pFile,"%lf",ptr);
            ptr++;
        }
    }
    else
    {
        cout <<"impossibile aprire file" << endl;
        return -1;
    }
    return 0;
}

Mat plot_histogram(float* istogramma)
{
    Mat white(200,256,CV_8UC1, 255);
    int i,j,c;
    float max = 0;

    for(j=0;j<256;j++)
    {
        if(max<istogramma[j])
            max=istogramma[j];
    }
    float scale = 190/max;

    for(j=0;j<256;j++)
    {
        c=(int)floor(scale*istogramma[j]);

        for(i=0;i<200;i++)
        {
            if(i<(200-c))
                white.at<unsigned char>(i,j)=255;
            else
                white.at<unsigned char>(i,j)=0;
        };
    };
    return white;
}


/*
 * This method is used to load the the historgams of the distributions.
 */
int histc(double *distribution_source, int n_samples, double step, double *hist_distribution)
{
    double x[51]={};
    x[0]=0;
    for(int i=1 ; i< 51 ; i++)
        x[i]= x[i-1]+step;

    for(int j=0; j<51; j++)
        for(int i=0 ; i< n_samples; i++)
        {
            if(x[j]<=distribution_source[i] && distribution_source[i] < x[j+1])
                hist_distribution[j]++;
        }

    for(int i=0 ; i< n_samples; i++)
    {
        if(x[50]==distribution_source[i])
            hist_distribution[50]++;
    }

    for(int i=0 ; i< 51; i++)
    {
        hist_distribution[i]/=n_samples;
    }

    return 0;
}

// This set of functions are used to get equalized image
float* cumulative(float* histogram)
{
    int i,j;
    float* z;
    z=new float [256];
    for(i=0;i<256;i++)
    {
        z[i]=0;
        for(j=0;j<=i;j++)
        {
            z[i]+=histogram[j];
        }
        //cout<< z[i]<<endl;
    }
    return z;
}



float* histogram_gray (Mat img)
{
    int i,j,k;
    float* z;
    z=new float [256];
    for(i=0;i<256;i++)
    {
        z[i]=0;
    }
    for(i=0;i<img.rows;i++)
    {
        for(j=0;j<img.cols;j++)
        {
            k = (img).at<unsigned char>(i,j);
            z[k]++;
        }
    }
    for(i=0;i<256;i++)
    {
        z[i]/=(float)(img.cols*img.rows);
    }
    return z;
}

