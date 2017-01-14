#include "histogram.hpp"
#include "utils.hpp"

vector<double> ComputeEHOG(Mat source,  vector<vector<Point> > regions);
vector<double> ComputePD(Mat source,  vector<vector<Point> > regions);
vector<double> ComputeStrokeWidth(Mat source,  vector<vector<Point> > regions);
/*
 * This method is used to find a value of the feature EHOG associated to each region. The working principle is to find edges in each region and
 * compute the angle of these pixels; the angle vaues are subdivided into four orientations groups and put toghether using a formula that takes into account
 * the main features of a letter shape.
 */
vector<double> ComputeEHOG(Mat source,  vector<vector<Point> > regions)
{
    //Source è RGB
    Mat grayscale(source.size(),CV_8UC1);
    cvtColor(source, grayscale, CV_RGB2GRAY);

    Mat kernel_gradient_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat kernel_gradient_y = (Mat_<float>(3,3) << 1,2,1,0,0,0,-1,-2,-1);

    Mat gradient_x=gradient_x.zeros(source.size(), CV_32F);
    Mat gradient_y=gradient_y.zeros(source.size(), CV_32F);
    Mat angleMap=angleMap.zeros(source.size(), CV_32F);
    Mat XY=XY.zeros(source.size(), CV_32F);
    Mat indexMap=indexMap.zeros(source.size(), CV_32F);

    int nAngle = 360/4;
    filter2D(grayscale , gradient_x, CV_32F, kernel_gradient_x);
    filter2D(grayscale, gradient_y, CV_32F, kernel_gradient_y);

    //Sobel(grayscale, gradient_x, CV_32F, 1, 0, 3);
    //Sobel(grayscale, gradient_y, CV_32F, 0, 1, 3);

    for(int rows = 0; rows < angleMap.rows; rows++)
    {
        for(int cols = 0; cols < angleMap.cols; cols++)
        {
            if(gradient_x.at<float>(rows,cols)==0) gradient_x.at<float>(rows,cols)=exp(-5);

            //if angle is 180 (atan a da -pi/2 a pi/2)
            //XY.at<float>(rows,cols)=(gradient_y.at<float>(rows,cols)/gradient_x.at<float>(rows,cols));
            //angleMap.at<float>(rows,cols) = (atan((double)(XY.at<float>(rows,cols))+(M_PI/2))*180)/M_PI;

            //if angle is 360 (atan2 va da -pi a pi)
            angleMap.at<float>(rows,cols) = ((atan2(gradient_y.at<float>(rows,cols), gradient_x.at<float>(rows,cols))+M_PI)*180)/M_PI;

            //cout<< "x "<<gradient_x.at<float>(rows,cols)<< "  y "<<gradient_y.at<float>(rows,cols)<<endl;
            //cout<<angleMap.at<float>(rows,cols)<<endl;
            angleMap.at<float>(rows,cols) += 45;

            if(angleMap.at<float>(rows,cols)>=360) angleMap.at<float>(rows,cols)-=360;
            indexMap.at<float>(rows,cols) = floor(((angleMap.at<float>(rows,cols)-exp(-5))/nAngle)+1);
            //cout<<indexMap.at<float>(rows,cols)<<endl;

        }
    }

    vector<double> result (regions.size());
    fill(result.begin(), result.end(), 0);

    Mat temp(grayscale.size(), CV_8UC1);

    for(int i=0; i<regions.size();i++)
    {
        temp = temp.zeros(grayscale.size(), CV_8UC1);

        for(unsigned int k=0; k < regions[i].size() ; k++)
        {
            temp.at<unsigned char>(regions[i][k].y,regions[i][k].x) = 255;
        }
        Canny(temp, temp, 50, 200);
        //phase(temp.rows, temp.cols, orientation);

        int w1=0,w2=0,w3=0,w4=0,w=0;
        for(unsigned int k=0; k < regions[i].size() ; k++)
        {
            //if(temp.at<unsigned int>(regions[i][k].y,regions[i][k].x) != 0)
            {
                int angle = (int)indexMap.at<float>(regions[i][k].y,regions[i][k].x);
                //cout<<angle<<endl;
                //if((angle>0 & angle<=45) || (angle>315 & angle<=360)) //Type 1
                if(angle==1)
                    w1++;
                //else if(angle>45 & angle<=135) //Type 2
                else if (angle==2)
                    w2++;
                //else if(angle>135 & angle<=225) //Type 3
                else if (angle==3)
                    w3++;
                //else if(angle>225 & angle<=315) //Type 4
                else if (angle==4 || angle==0)
                    w4++;
            }

        }

        w=0 ;
        w=w1+w2+w3+w4;
        //cout<< "w " << w << " w1 " << w1 << " w2 " << w2 << " w3 " << w3 << " w4 " << w4 << endl;
        if(w==0) result[i]=-1;
        else result[i] = sqrt(pow((w1-w3),2)+pow((w2-w4),2))/w;
        //cout<< (double) (result[i])<<endl;
    }
    return result;
}

/*
 * This method is used to compute the perceptual divergence of an image from the analysis of a single region;
 * It returns a vector containing in each position the value of the pd of the region, according to the color divergence
 * in the region, found from the histogram.
 */
vector<double> ComputePD(Mat source,  vector<vector<Point> > regions)
{
    vector<double> result (regions.size());
    fill(result.begin(), result.end(), 0);

    //vector<Mat> channels;
    Mat channels[3];
    split(source, channels);

    int controllo_sgamato = 0;
    for (unsigned int i= 0 ; i< regions.size(); i++)
    {
        float KLD[3]={};

        Rect rettangolo = boundingRect(regions[i]);
        rettangolo.x -= 1;
        rettangolo.y -= 1;
        rettangolo.height += 2;
        rettangolo.width += 2;

        //Checks on box dimension
        if(rettangolo.x < 0)
        {
            rettangolo.x = 0;
            //cout << "Bounding box modificato!\n";
            controllo_sgamato++;
            //cout << "il contatore vale:  " << controllo_sgamato << endl;
        }
        if(rettangolo.x + rettangolo.width > source.cols)
        {
            rettangolo.width = source.cols - rettangolo.x;
            //cout << "Bounding box modificato!\n";
            controllo_sgamato++;
        }
        if(rettangolo.y < 0)
        {
            rettangolo.y = 0;
            //cout << "Bounding box modificato!\n";
            controllo_sgamato++;
        }
        if(rettangolo.y + rettangolo.height > source.rows)
        {
            rettangolo.height = source.rows - rettangolo.y;
            //cout << "Bounding box modificato!\n";
            controllo_sgamato++;
        }

        // list of r* elements
        vector<Point> rect_star_element(0);
        rect_star_element.clear();

        // create a black image
        Mat out = out.zeros(source.size(), CV_8UC1);
        // plot a white box
        rectangle(out,rettangolo, Scalar(255),CV_FILLED, 8,0);
        // set region's point to zero
            for(unsigned int k=0; k < regions[i].size() ; k++)
            {
                out.at<unsigned char>(regions[i][k].y,regions[i][k].x) = 0;
            }

        // put in r* all white points
        for(int w=rettangolo.x; w< rettangolo.x + rettangolo.width; w++)
        {
            for(int h=rettangolo.y; h< rettangolo.y + rettangolo.height; h++)
            {
                if (out.at<unsigned char>(h,w) == 255)
                {
                rect_star_element.push_back(Point(w,h));
                }
            }
        }

        // Iterate for 3 channels
        for(int chan = 0; chan < 3; chan ++)
        {
            // create histogram of colors
            vector<float> histFore(26);
            vector<float> histBack(26);
            fill(histFore.begin(), histFore.end(), 0);
            fill(histBack.begin(), histBack.end(), 0);

            // Definition of scale for histogram
            int x[26]={};
            x[0]=5;
            for(int s=1 ; s< 26 ; s++)
            {
                x[s]= x[s-1]+10;
            }

            // Create hist foreground
            for(unsigned int j=0 ; j < regions[i].size(); j++)
            {
                if(channels[chan].at<unsigned char>(regions[i][j].y,regions[i][j].x) < x[0])
                    histFore[0]++;
                else
                {
                    for(int k = 1; k < 26; k++)
                        if (x[k-1] < channels[chan].at<unsigned char>(regions[i][j].y,regions[i][j].x)
                                && channels[chan].at<unsigned char>(regions[i][j].y,regions[i][j].x) <= x[k])
                            histFore[k]++;
                }
            }
            // Normalize
            for(int j=0 ; j< 26; j++)
            {
                histFore[j] /= (double)regions[i].size();
            }

            // Create hist background
            for(unsigned int j=0 ; j < rect_star_element.size(); j++)
            {
                if(channels[chan].at<unsigned char>(rect_star_element[j].y, rect_star_element[j].x) < x[0])
                    histBack[0]++;
                else
                {
                    for(int k = 1; k < 26; k++)
                        if (x[k-1] < channels[chan].at<unsigned char>(rect_star_element[j].y, rect_star_element[j].x)
                                && channels[chan].at<unsigned char>(rect_star_element[j].y, rect_star_element[j].x) <= x[k])
                            histBack[k]++;
                }
            }

            // Normalize
            for(int j=0 ; j< 26; j++)
            {
                histBack[j] /= (float)rect_star_element.size();
            }

            //Check null elements
            for(int j=0 ; j< 26; j++)
            {
                if(histBack[j] == 0.0)
                    histBack[j] = 0.00001;
                if(histFore[j] == 0.0)
                    histFore[j] = 0.00001;
            }

            vector<float> KLD_distance(26);

            for(int j=0 ; j< 26; j++)
            {
                KLD_distance[j]=histFore[j]*log(histFore[j]/histBack[j]);
                KLD[chan] += KLD_distance[j];
            }

            out = Scalar(0);
            for(unsigned int k=0; k < rect_star_element.size() ; k++)
            {
                out.at<unsigned char>(rect_star_element[k].y, rect_star_element[k].x) = channels[chan].at<unsigned char>(rect_star_element[k].y, rect_star_element[k].x);
            }
        }
        result[i]= KLD[0] + KLD[1] + KLD[2];
    }

    //cout << "modifiche totali: " << controllo_sgamato << endl;
    for(unsigned int k=0; k < result.size() ; k++)
    {
        if (result[k]>=35) result[k]=49.9;
    }
    return result;
}

/*
 * This method is used to compute the stroke width of an image from the analysis of a single region;
 * It returns a vector containing in each position the value of the ratio between variance and the
 * mean square of the region
 */
vector<double> ComputeStrokeWidth(Mat source,  vector<vector<Point> > regions)
{
    //Initialization of result
    vector <double> result(regions.size());
    fill(result.begin(), result.end(), 0);

    Mat source_dist(source.size(), CV_32FC1);
    distanceTransform(source, source_dist,CV_DIST_L2, CV_DIST_MASK_PRECISE);
	//	imshow("source_dist", source_dist);
	//	waitKey();

    Mat immagine_regione(source.size(), CV_8UC1);
    Mat skeldist_regione(source.size(), CV_32FC1);

    for (unsigned int i=0 ; i< regions.size(); i++)
    {
        immagine_regione = immagine_regione.zeros(source.size(), CV_8UC1);
        skeldist_regione = skeldist_regione.zeros(source.size(), CV_32FC1);

        //Population of immagine_regione with Point contained in regions[i]
        for(unsigned int j=0; j < regions[i].size() ; j++)
        {
            immagine_regione.at<unsigned char>(regions[i][j].y,regions[i][j].x) = 255;
        }
        //imshow("immagine regione", immagine_regione);
        //waitKey();

        //Computation of the skeleton for the region
        Mat skeleton_region = skeleton_region.zeros(source.size(), CV_8UC1);
        scheletro(immagine_regione, &skeleton_region);
        //imshow("scheletro regione", skeleton_region);
        //waitKey();

        //Evaluation of skeleton size: if it is less than 5 pixels the result of the region is assigned
        //to -1 to indicate that will be deleted
        int number_non_zero = countNonZero(skeleton_region);
        //cout << "region n° " << i << "\tskel pixel: " << number_non_zero << endl;

        if(number_non_zero < 5)
        {
            result[i] = -1;
        }
        else
        {
            float media=0.0;
            float media_quad=0.0;
            int elem=0;

            skeleton_region.convertTo(skeleton_region,CV_32FC1);
            multiply(source_dist,skeleton_region,skeldist_regione,1);

            for (int k=0 ; k < skeldist_regione.rows; k++)
            {
                for(int j=0; j < skeldist_regione.cols; j++)
                {
                    if (skeldist_regione.at<float> (k,j) > 0.0)
                    {
                        media += skeldist_regione.at<float> (k,j);
                        media_quad += pow(skeldist_regione.at<float> (k,j),2);
                        elem++;
                    }
                }
            }

            media = media/(float)elem;
            media_quad = media_quad/(float)elem;
            result[i]= (media_quad - pow(media,2))/(float)(pow(media,2));

            //Checking if the element is >= 2
            if(result[i] >= 2)
                result[i] = 1.99;
        }
        //cout << result[i] << endl;
        //waitKey();
    }
    return result;
}



/*
 * This method is used to get the results for each cue described. In particulat it gets as input the result of the
 * function which computes the value of the cue feature associated to each region and returns a vector, of the same lenght of the
 * regions vector, where only the detected regions are set; this process is performed according to the positive and negative distributions
 * of each cue (previously scaling the results between 0 and 50). An image is shown where only the set regions are plotted in white on a black background.
 */
vector<int> regions_cue(vector<double> cue, double positive_dist[51], double negative_dist[51], Mat source, vector<vector<Point> > regions, int numero, Mat *risultato)
{
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    //cout << "Regioni cue prima " << cue.size()<<endl;

    vector<int> ris(cue.size());
    minMaxLoc( cue, &minVal, &maxVal, &minLoc, &maxLoc );
    //cout<<"max"<<maxVal<<endl;
    //cout<<"min"<<minVal<<endl;
    //cout << "Regioni cue dopo " <<cue.size()<<endl;

	//    for(int index = 0; index < cue.size(); index++)
	//        cout << "Contenuto di cue in posizione " << index << " \t" << cue[index] << endl;

    // Normalize cue between 0 and 1
    for(unsigned int i=0; i<cue.size();i++)
    {
        cue[i]=cue[i]/maxVal;
    }

    for(unsigned int i=0; i<cue.size();i++)
    {
        double positive, negative;

        int val = (int)floor(50*cue[i]);

        positive=positive_dist[val];
        negative=negative_dist[val];

        if(positive>negative)
            ris[i]=1;
        else
            ris[i]=2;
    }

    Mat uscita = uscita.zeros(source.size(), CV_8U);
    for(unsigned int j=0; j<regions.size(); j++)
    {
        for(unsigned int k=0; k < regions[j].size() ; k++)
        {
            if(ris[j]==1)
                uscita.at<unsigned char>(regions[j][k].y,regions[j][k].x) = 255;
        }
    }

	//    namedWindow("regioni", WINDOW_NORMAL);
	//    imshow("regioni", uscita);
	//    waitKey(0);

    if (numero < 100)
    {
        stringstream ss;
        ss << "./testing2003/testing_cues2003/00" << numero << "" <<".png";
        imwrite(ss.str(),uscita);
    }
    if (100 <=numero && numero < 1000)
    {
        stringstream ss;
        ss << "./testing2003/testing_cues2003/0" << numero << "" <<".png";
        imwrite(ss.str(),uscita);
    }
    if (1000 <=numero && numero < 10000)
    {
        stringstream ss;
        ss << "./testing2003/testing_cues2003/" << numero << "" <<".png";
        imwrite(ss.str(),uscita);
    }

    for(unsigned int j=0; j<uscita.rows; j++)
    {
        for(unsigned int k=0; k < uscita.cols ; k++)
        {
            risultato->at<unsigned char>(j,k)=uscita.at<unsigned char>(j,k);
        }
    }
    return ris;
}



/*
 * This method is used to get the final results by putting toghether the three cues partial results using the Bayes probability theorem.
 * Each cue result is normalized and re-scaled between 0 and 50 in order to use the positive and negative distributions; two a priori probability values
 * of text and non text regions are used and set to 0.7 and 0.3.
 */
Mat Bayes(double positive_dist_SW[51], double negative_dist_SW[51], double positive_dist_PD[51], double negative_dist_PD[51], double positive_dist_EHOG[51], double negative_dist_EHOG[51], vector<double> SW, vector<double> PD, vector<double> EHOG, Mat source, vector<vector<Point> > regions)
{
    Mat result(source.size(),CV_8U);
    result = result.zeros(source.size(), CV_8U);

    //Dimensions check
    if(SW.size()!=PD.size() || EHOG.size()!=regions.size() || SW.size()!=EHOG.size() || SW.size()!=regions.size() || EHOG.size()!=
       PD.size() || PD.size()!=regions.size())
        cout<<"dimensions error!"<<endl;

	//    for(unsigned int i=0; i<regions.size();i++)
	//    {
	//        if(SW[i]<0 || PD[i]<0 || EHOG[i]<0)     regions.erase(regions.begin()+i);
	//    }

	//    for (unsigned int i=0 ; i< SW.size(); i++)
	//        cout << "contenuto bright PD: "<< i << " " << SW[i] << endl;

    //Put all values between 0 and 50 (distribution bins)
    double minValSW, maxValSW;
    Point minLocSW, maxLocSW;
    minMaxLoc( SW, &minValSW, &maxValSW, &minLocSW, &maxLocSW );

    double minValPD, maxValPD;
    Point minLocPD, maxLocPD;
    minMaxLoc( PD, &minValPD, &maxValPD, &minLocPD, &maxLocPD );

    double minValEHOG, maxValEHOG;
    Point minLocEHOG, maxLocEHOG;
    minMaxLoc( EHOG, &minValEHOG, &maxValEHOG, &minLocEHOG, &maxLocEHOG );

    for(unsigned int i=0; i<regions.size();i++)
    {
        SW[i]=SW[i]/maxValSW;
        PD[i]=PD[i]/maxValPD;
        EHOG[i]=EHOG[i]/maxValEHOG;

        SW[i] = floor(50*SW[i]);
        PD[i] = floor(50*PD[i]);
        EHOG[i] = floor(50*EHOG[i]);
    }

    // Code related to Bayes modificato
    /*
    float dimensione_regione = 0.0;
    float media_regioni = 0.0;
    int scelta = 0;
    for(unsigned int i=0; i<regions.size();i++)
        dimensione_regione +=regions[i].size();
    media_regioni = dimensione_regione/regions.size();

    //cout << "Dimensione immagine: " << dimensione_regione << "\nMedia regioni: " << media_regioni << endl;

    float risultato = media_regioni/dimensione_regione;
    if(risultato>0.017)
        scelta = 0;
    else
        scelta = 1;

    //cout << "Rapporto: " << risultato << " quindi scelta:" << scelta << endl;
    */

    double BayesP, BayesN, Bayes;
    for(int i=0;i<regions.size();i++)
    {
        Bayes = 0;  BayesP = 0;  BayesN = 0;
        if(SW[i]<0 || PD[i]<0 || EHOG[i]<0)
            cout<<"regione scartata (-1)"<<endl;
        else
        {
            BayesP = prioriT*(positive_dist_EHOG[(int)EHOG[i]]*positive_dist_PD[(int)PD[i]]*positive_dist_SW[(int)SW[i]]);
            BayesN = prioriN*(negative_dist_EHOG[(int)EHOG[i]]*negative_dist_PD[(int)PD[i]]*negative_dist_SW[(int)SW[i]]);

            // Related to Bayes modificato
            /*
            if(scelta == 1)
            {
                BayesP = prioriT*(positive_dist_EHOG[(int)EHOG[i]]*positive_dist_PD[(int)PD[i]]);
                BayesN = prioriN*(negative_dist_PD[(int)PD[i]]*negative_dist_EHOG[(int)EHOG[i]]);
            }
            else
            {
                BayesP = prioriT*(positive_dist_PD[(int)PD[i]]*positive_dist_SW[(int)SW[i]]);
                BayesN = prioriN*(negative_dist_PD[(int)PD[i]]*negative_dist_SW[(int)SW[i]]);
            }
            */
            if(BayesN==0 & BayesP==0)
                cout<<"regione scartata (nan)"<<endl;
            else
            {
                Bayes=BayesP/(BayesP+BayesN);
                //cout<<Bayes<<endl;
                if(Bayes>=0.5)
                    for(unsigned int k=0; k < regions[i].size() ; k++)
                    {
                        result.at<unsigned char>(regions[i][k].y,regions[i][k].x) = 255;
                    }
            }
        }
    }

    namedWindow("bayes", WINDOW_NORMAL);
    imshow("bayes", result);
    waitKey(0);
    return result;
}

