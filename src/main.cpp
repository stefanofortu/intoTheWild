#include "config.hpp"
#include "cues.hpp"
#include "mserWrap.hpp"
#include "utils.hpp"
#include "histogram.hpp"


using namespace cv ;
using namespace std;

int main()
{
    //Loading of distributions
    double distribution_SW_P[51], distribution_SW_N[51]={};
    double data_SWV_P[POSITIVE_SAMPLES]={};
    double data_SWV_N[NEGATIVE_SAMPLES]={};
    caricaDistribuzione("./distribution/CueSWVimP.txt",data_SWV_P);
    caricaDistribuzione("./distribution/CueSWVimN.txt",data_SWV_N);
    histc(data_SWV_P, POSITIVE_SAMPLES, 0.04, distribution_SW_P);
    histc(data_SWV_N, NEGATIVE_SAMPLES, 0.04, distribution_SW_N);

    double distribution_HOG_P[51], distribution_HOG_N[51]={};
    double data_HOG_P[POSITIVE_SAMPLES]={};
    double data_HOG_N[NEGATIVE_SAMPLES]={};
    caricaDistribuzione("./distribution/CueHOGimP.txt",data_HOG_P);
    caricaDistribuzione("./distribution/CueHOGimN.txt",data_HOG_N);
    histc(data_HOG_P, POSITIVE_SAMPLES, 0.02, distribution_HOG_P);
    histc(data_HOG_N, NEGATIVE_SAMPLES, 0.02, distribution_HOG_N);

    double distribution_PD_P[51], distribution_PD_N[51]={};
    double data_PD_P [POSITIVE_SAMPLES]={};
    double data_PD_N [NEGATIVE_SAMPLES]={};
    caricaDistribuzione("./distribution/CuePDimP.txt",data_PD_P);
    caricaDistribuzione("./distribution/CuePDimN.txt",data_PD_N);
    histc(data_PD_P, POSITIVE_SAMPLES, 0.7, distribution_PD_P);
    histc(data_PD_N, NEGATIVE_SAMPLES, 0.7, distribution_PD_N);

    int numero = 1;
    //for(int immagine = 1; immagine < 89; immagine ++)
    for(int immagine = 1; immagine < 2; immagine ++)
    {
        stringstream percorso;
        //percorso << "../img/OrientedSceneTextDataset/Proof (" << immagine << ").jpg";
        percorso << "../img/150.jpg";
        cout << percorso.str() << endl;
        Mat source = imread(percorso.str(), CV_LOAD_IMAGE_COLOR);
        //Mat source = imread("150.jpg", CV_LOAD_IMAGE_COLOR);

        // Check for invalid input
        if(!source.data )
        {
            cout <<  "Could not open or find the image" << endl ;
            return -1;
        }
        cout << "Image is open \n\n" ;
        //    namedWindow("source",WINDOW_NORMAL);
        //    imshow("source",source);

        // Equalization part
        Mat channels[3];
        split(source, channels);

        channels[0]=equalized(channels[0]);
        channels[1]=equalized(channels[1]);
        channels[2]=equalized(channels[2]);

        //    float* v0 = histogram_gray (channels[0]);
        //    float* v1 = histogram_gray (channels[1]);
        //    float* v2 = histogram_gray (channels[2]);

        //    Mat z0 = plot_histogram(v0);
        //    Mat z1 = plot_histogram(v1);
        //    Mat z2 = plot_histogram(v2);

        merge(channels, 3, source); //End of equalization

        Mat brightGrayImg(source.rows, source.cols,CV_8U);
        Mat darkGrayImg(source.rows,source.cols, CV_8U);

        PreEMSER(source , &brightGrayImg, &darkGrayImg, false);
        //    imshow("brightGrayImg", brightGrayImg);
        //    imshow("darkGrayImg", darkGrayImg);

        //Definition of regions which will contain in each position a vector containing itself the set of
        //points that belong to that region
        vector<vector<Point> > regions_b;
        vector<vector<Point> > regions_d;

        Mat bright_MSER = bright_MSER.zeros(brightGrayImg.size(), brightGrayImg.type());
        vl_feat_mser (brightGrayImg, regions_b,&bright_MSER ,MSER_BRIGHT_ON_DARK);

        Mat dark_MSER = dark_MSER.zeros(darkGrayImg.size(), darkGrayImg.type());
        //regions_d  = calcolaMSER(darkGrayImg,0,&dark_MSER);
        vl_feat_mser (darkGrayImg, regions_d, &dark_MSER ,MSER_DARK_ON_BRIGHT);

        //    imshow("bright MSER", bright_MSER);
        //    imshow("dark MSER", dark_MSER);

        // Stroke width method
        vector<double> CueSW_d, CueSW_b;
        CueSW_b = ComputeStrokeWidth(bright_MSER, regions_b);
        CueSW_d = ComputeStrokeWidth(dark_MSER, regions_d);
        cout << "Calcolo Stroke width terminato" << endl;
        //cout << "Le regioni sono: "<<CueSW_b.size() << " e "<< CueSW_d.size() << endl;

        // Perceptual divergence method
        vector<double> CuePD_b, CuePD_d;
        CuePD_b = ComputePD(source, regions_b);
        CuePD_d = ComputePD(source, regions_d);
        cout << "Calcolo PD terminato" << endl;

        // eHOG method
        vector<double> EHOG_d(regions_d.size());
        vector<double> EHOG_b(regions_b.size());
        EHOG_d = ComputeEHOG(source, regions_d );
        EHOG_b = ComputeEHOG(source, regions_b );
        cout << "Calcolo eHOG  terminato" << endl;

        // Checking for elements that must be deleted
        regions_b = deleting_elements(CueSW_b, CuePD_b, EHOG_b, regions_b);
        regions_d = deleting_elements(CueSW_d, CuePD_d, EHOG_d, regions_d);

        // Normalization of result to be comparable with histograms
        vector<int> risehog_b(EHOG_b.size());
        vector<int> risehog_d(EHOG_d.size());
        vector<int> rissw_b(CueSW_b.size());
        vector<int> rissw_d(CueSW_d.size());
        vector<int> rispd_b(CueSW_b.size());
        vector<int> rispd_d(CueSW_d.size());

        Mat SW_result_b = SW_result_b.zeros(source.size(), CV_8UC1);
        Mat SW_result_d = SW_result_d.zeros(source.size(), CV_8UC1);
        Mat PD_result_b = PD_result_b.zeros(source.size(), CV_8UC1);
        Mat PD_result_d = PD_result_d.zeros(source.size(), CV_8UC1);
        Mat EHOG_result_b = EHOG_result_b.zeros(source.size(), CV_8UC1);
        Mat EHOG_result_d = EHOG_result_d.zeros(source.size(), CV_8UC1);

        //Application of algorithm to obtain the result for each method (both for bright and dark)
        numero = immagine*10;
        risehog_b = regions_cue(EHOG_b, distribution_HOG_P, distribution_HOG_N, source, regions_b,numero,&EHOG_result_b);
        numero++;
        //    namedWindow("Ehog result", WINDOW_NORMAL);
        //    imshow("Ehog result", EHOG_result_b);
        //    waitKey();

        risehog_d = regions_cue(EHOG_d,distribution_HOG_P,distribution_HOG_N,source,regions_d,numero,&EHOG_result_d);
        numero++;
        rissw_b = regions_cue(CueSW_b,distribution_SW_P,distribution_SW_N,source,regions_b,numero,&SW_result_b);
        numero++;
        rissw_d = regions_cue(CueSW_d,distribution_SW_P,distribution_SW_N,source,regions_d,numero,&SW_result_d);
        numero++;
        rispd_b = regions_cue(CuePD_b,distribution_PD_P,distribution_PD_N,source,regions_b,numero,&PD_result_b);
        numero++;
        rispd_d = regions_cue(CuePD_d,distribution_PD_P,distribution_PD_N,source,regions_d,numero,&PD_result_d);
        numero++;

        vector<Mat> unione_b, unione_d;
        //unione_b[0] = PD_result_b;
        unione_b.push_back(PD_result_b);
        unione_b.push_back(EHOG_result_b);
        unione_b.push_back(SW_result_b);

        unione_d.push_back(PD_result_d);
        unione_d.push_back(EHOG_result_d);
        unione_d.push_back(SW_result_d);

        Mat colour_b = colour_b.zeros(source.size(), CV_8UC3);
        merge(unione_b,colour_b);
        //    namedWindow("Colori bright", WINDOW_NORMAL);
        //    imshow("Colori bright", colour_b);
        //    waitKey();

        Mat colour_d = colour_d.zeros(source.size(), CV_8UC3);
        merge(unione_d,colour_d);
        //    namedWindow("Colori dark", WINDOW_NORMAL);
        //    imshow("Colori dark", colour_d);
        //    waitKey();

        stringstream ss2,ss3;
        ss2 << "./../results/" << immagine << "colori_b.png";
        ss3 << "./../results/" << immagine << "colori_d.png";
        //    ss2 << "./testing2003/colori_b.png";
        //    ss3 << "./testing2003/colori_d.png";

        imwrite(ss2.str(),colour_b);
        imwrite(ss3.str(),colour_d);

        //Application of Bayes
        Mat bayes_b(source.size(),CV_8U);
        bayes_b=Bayes(distribution_SW_P, distribution_SW_N, distribution_PD_P, distribution_PD_N, distribution_HOG_P, distribution_HOG_N, CueSW_b, CuePD_b, EHOG_b, source, regions_b);

        Mat bayes_d(source.size(),CV_8U);
        bayes_d=Bayes(distribution_SW_P, distribution_SW_N, distribution_PD_P, distribution_PD_N, distribution_HOG_P, distribution_HOG_N, CueSW_d, CuePD_d, EHOG_d, source, regions_d);

        //Saving the Bayes's results
        stringstream ss,ss1;
        ss << "./../results/" << immagine << "_b.png";
        ss1 << "./../results/" << immagine << "_d.png";

        imwrite(ss.str(),bayes_b);
        imwrite(ss1.str(),bayes_d);

        cout << "Ho analizzato l'immagine: " << immagine <<endl;
        //waitKey();
    }
    return 0;
}
