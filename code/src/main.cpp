#include "funzioni.hpp"

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

//  This second main is used to compute precision and recall
//int main()
//{
//    int numero = 0;
////    float precision_dark[88];
////    float precision_bright[88];
////    float recall_dark[88];
////    float recall_bright[88];
//    float precision[176];
//    float recall[176];

//    //for(int immagine = 1; immagine < 89; immagine ++)
//    for(int immagine = 1; immagine < 89; immagine ++)
//    {
//        // Loading Bayes's results
//        stringstream percorso_b, percorso_d;
//        percorso_b << "./testingOrientedScene/"<< immagine <<"_b.png";
//        percorso_d << "./testingOrientedScene/"<< immagine <<"_d.png";

//        // Bright on dark
//        Mat bayes_b = imread(percorso_b.str(), CV_LOAD_IMAGE_GRAYSCALE);
//        if(!bayes_b.data )
//        {
//            cout <<  "Could not open or find the image" << endl ;
//            return -1;
//        }
//        cout << "Image is open \n" ;

//        // Dark on bright
//        Mat bayes_d = imread(percorso_d.str(), CV_LOAD_IMAGE_GRAYSCALE);
//        if(!bayes_d.data )
//        {
//            cout <<  "Could not open or find the image" << endl ;
//            return -1;
//        }
//        cout << "Image is open \n" ;

//        // Check of union of bayes
//        Mat somma_bayes = bayes_d.zeros(bayes_d.rows, bayes_d.cols, bayes_d.type());
//        bitwise_or(bayes_b, bayes_d, somma_bayes);
////        namedWindow( "somma", CV_WINDOW_AUTOSIZE );
////        imshow( "somma", somma_bayes);
////        waitKey(0);

//        stringstream ss;
//        ss << "./testingOrientedSceneUnionBayes/" << immagine << ".png";
//        imwrite(ss.str(),somma_bayes);

//        // Load Groundtruth
//        stringstream percorso_ground;
//        percorso_ground << "./maps/Proof_GT (" << immagine <<").bmp";
//        Mat groundtruth = imread(percorso_ground.str(), CV_LOAD_IMAGE_GRAYSCALE);
//        if(!groundtruth.data )
//        {
//            cout <<  "Could not open or find the image" << endl ;
//            return -1;
//        }
//        cout << "Image is open \n" ;

//        // Definition of variables to count elements equal to one (since the three Mat are binary)
//        int gt = 0;
////        int noi_d = 0;
////        int noi_b = 0;
//        int noi_sum = 0;

//        // Computation of elements equal to one
//        gt=countNonZero(groundtruth);
////        noi_d=countNonZero(bayes_d);
////        noi_b=countNonZero(bayes_b);
//        noi_sum = countNonZero(somma_bayes);

//        // Checking of elements to avoid nan during computation of precision and recall
//        if(gt == 0)
//            gt = gt+1;
////        if(noi_b == 0)
////            noi_b = noi_b+1;
////        if(noi_d == 0)
////            noi_d = noi_d+1;
//        if(noi_sum == 0)
//            noi_sum = noi_sum+1;

//        // Computation of true positive in the bayes results
////        bitwise_and(bayes_d, groundtruth, bayes_d);
////        bitwise_and(bayes_b, groundtruth, bayes_b);
//        bitwise_and(somma_bayes, groundtruth, somma_bayes);

////        namedWindow( "b_d", CV_WINDOW_AUTOSIZE );
////        imshow( "b_d", bayes_d);
////        waitKey(0);
////        namedWindow( "b_b", CV_WINDOW_AUTOSIZE );
////        imshow( "b_b", bayes_b );
////        waitKey(0);

//        // Definition of variables to count elementes equals to one (true positive)
////        int match_d = 0;
////        int match_b = 0;
//        int match_sum = 0;

//        // Computation of elements equal to one
////        match_d=countNonZero(bayes_d);
////        match_b=countNonZero(bayes_b);
//        match_sum = countNonZero(somma_bayes);
//        //cout<<match_b<<endl<<match_d<<endl;

//        // Definition of variables to store precision and recall
////        float precision_d = 0.0;
////        float recall_d = 0.0;
////        float precision_b = 0.0;
////        float recall_b = 0.0;
//        float precision_sum = 0.0;
//        float recall_sum = 0.0;

//        // Computation of precision and recall
////        precision_d=(float)match_d/(float)gt;
////        recall_d=(float)match_d/(float)noi_d;
////        precision_b=(float)match_b/(float)gt;
////        recall_b=(float)match_b/(float)noi_b;
//        precision_sum=(float)match_sum/(float)gt;
//        recall_sum=(float)match_sum/(float)noi_sum;

//        cout << fixed;
////        cout<<"Image "<< immagine << " provides:\tPrecision bright-vs-dark "<< precision_b<<"\t\tRecall bright-vs-dark "<<recall_b<<endl;
////        cout<<"Image "<< immagine << " provides:\tPrecision dark-vs-bright "<< precision_d<<"\t\tRecall dark-vs-bright "<<recall_d<<endl;
//        cout<<"Image "<< immagine << " provides:\tPrecision "<< precision_sum<<"\t\tRecall "<<recall_sum<<endl;
//        cout << endl;

//        // Storing results into their vectors (Note: bright and then dark)
////        precision[numero] = precision_b;
////        recall[numero] = recall_b;
////        numero++;
////        precision[numero] = precision_d;
////        recall[numero] = recall_d;
////        numero++;
//        precision[numero] = precision_sum;
//        recall[numero] = recall_sum;
//        numero++;
//    }

//    for(int passo = 0; passo < 88; passo ++)
//    {
//        cout << fixed;
//        //cout << "Precision " << precision[passo] << "\t Recall " << recall[passo] << endl;
//        cout << precision[passo] << endl;
//    }
//    return 0;
//}

