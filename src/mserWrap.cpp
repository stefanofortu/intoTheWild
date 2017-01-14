
#include "mserWrap.hpp"
#include "histogram.hpp"
#include "utils.hpp"

int PreEMSER(Mat source,Mat* brightGrayImg_OUT, Mat*darkGrayImg_OUT, bool show);
int vl_feat_mser(Mat input_img, vector <vector <Point > > & regioni_FF , Mat* output_img ,int mser_type);


/*
 * This method is used to process the image before applying the MSER method.
 * It smooths the image using a guided filter and extract the dark-on-bright version and the bright-on-dark using the
* hog_original funciont (so the gradient).
 */

int PreEMSER(Mat source,Mat* brightGrayImg_OUT, Mat*darkGrayImg_OUT, bool show)
{
    Mat grayscale;
    cvtColor(source, grayscale, CV_RGB2GRAY);
    if (show == true)
    {
        namedWindow("RGB", CV_WINDOW_AUTOSIZE);
        imshow("RGB", source);
        waitKey(0);
        namedWindow("HSI", CV_WINDOW_AUTOSIZE);
        imshow("HSI", grayscale);
        imwrite("GRAYSCALE.png",grayscale);
    }

    Mat filtered;
    bilateralFilter(grayscale, filtered, 5, 50, 50);

    if (show == true)
    {
        namedWindow("Filtered (bilateral)", CV_WINDOW_AUTOSIZE);
        imshow("Filtered (bilateral)", filtered);
        waitKey(0);
    }

    Mat hog;
    hog = hog_original(filtered);

    Mat brightGrayImg(source.rows,source.cols, hog.type());
    Mat darkGrayImg(source.rows,source.cols, hog.type());

    grayscale.convertTo(grayscale, hog.type());

    for (int rows=0; rows<hog.rows; rows++)
        for (int cols=0; cols<hog.cols; cols++)
        {
            brightGrayImg.at<float>(rows,cols)=grayscale.at<float>(rows,cols)-0.5*hog.at<float>(rows,cols);
            darkGrayImg.at<float>(rows,cols)=grayscale.at<float>(rows,cols)+0.5*hog.at<float>(rows,cols);
        }

    brightGrayImg.convertTo(brightGrayImg, CV_8U);
    darkGrayImg.convertTo(darkGrayImg, CV_8U);

    if (show == true){
        namedWindow("Dark", CV_WINDOW_AUTOSIZE);
        imshow("Dark", brightGrayImg);
        imwrite("Dark.png", brightGrayImg);
        waitKey(0);
        namedWindow("Bright", CV_WINDOW_AUTOSIZE);
        imshow("Bright", darkGrayImg);
        imwrite("Bright.png", darkGrayImg);
        waitKey(0);
        waitKey(0);
    }

    *brightGrayImg_OUT = brightGrayImg;
    *darkGrayImg_OUT   = darkGrayImg;
    return 0;
}


/*
 * This method is used to compute the MSER method using the VLFEAT library.
 */
int vl_feat_mser (Mat input_img, vector <vector <Point > > & regioni_FF , Mat* output_img ,int mser_type)
{
    int      bright_on_dark;
    int      dark_on_bright;
    int      verbose = 0 ;
    enum    {ndims = 2} ;
    int     dims [ndims] ;
    dims[0] = input_img.cols ;
    dims[1] = input_img.rows ;
    double   delta         = 10 ;
    double   max_area      = 0.1 ;
    double   min_area      = 0.00002 ;
    double   max_variation = 0.2 ;
    double   min_diversity = 0.5 ;

    VlMserFilt      *filt = 0 ;
    VlMserFilt      *filtinv = 0;
    vl_uint8        *data = 0 ;
    vl_uint const   *regions ;
    int              nregions = 0;

    switch (mser_type)
    {
        case 0 :
        {
            bright_on_dark = 1;
            dark_on_bright = 0;
            break;
        }
        case 1 :
        {
            bright_on_dark = 0;
            dark_on_bright = 1;
            break;
        }
        default :
        {
            cout << "Tipo di Mser non definito correttamente" << endl;
            return -1;
        }
    }

    data = (vl_uint8*) malloc(input_img.rows*input_img.cols*sizeof(vl_uint8));

    /// SET FILTER  ----------------------------------------------
    dims[0] = input_img.cols ;
    dims[1] = input_img.rows ;

    filt = vl_mser_new (ndims, dims) ;
    filtinv = vl_mser_new (ndims, dims) ;

    if (!filt || !filtinv)
    {
        cout << "Could not create an MSER filter. \n" ;
        return -1 ;
    }

    if (delta         >= 0) vl_mser_set_delta          (filt, (vl_mser_pix) delta) ;
    if (max_area      >= 0) vl_mser_set_max_area       (filt, max_area) ;
    if (min_area      >= 0) vl_mser_set_min_area       (filt, min_area) ;
    if (max_variation >= 0) vl_mser_set_max_variation  (filt, max_variation) ;
    if (min_diversity >= 0) vl_mser_set_min_diversity  (filt, min_diversity) ;

    if (verbose)
    {
        cout << "mser: parameters:\n" ;
        cout << "mser:   delta         = " <<  vl_mser_get_delta        (filt) << endl;
        cout << "mser:   max_area      = " << vl_mser_get_max_area      (filt) << endl;
        cout << "mser:   min_area      = " << vl_mser_get_min_area      (filt) << endl;
        cout << "mser:   max_variation = " << vl_mser_get_max_variation (filt) << endl;
        cout << "mser:   min_diversity = " << vl_mser_get_min_diversity (filt) << endl;
    }

    /// process the image-------------------------------------------------------
    if (bright_on_dark)
    {   //vl_mser_pix = vl_mser_pix = unsigned 8bit integer
        int k=0;
        for(int i=0; i<input_img.rows; i++)
        {
            for (int j=0; j<input_img.cols; j++)
            {
                data[k]=(vl_uint8) (255 - input_img.at<unsigned char>(i,j));
                k++;
            }
        }

        vl_mser_process (filt, (vl_mser_pix*) data) ;

        nregions=0;
        /* Save result */
        nregions = vl_mser_get_regions_num (filt) ;
        cout << "numero regioni trovate (Bright_On_Dark) : " << nregions << endl;
        regions  = vl_mser_get_regions     (filt) ;

        //conversion from seed_regions to seed_point
        vector < Point > seedPoints;
        seedPoints.clear();
        int x,y;
        for (int i = 0 ; i < nregions ; i++)
        {
            x = regions[i] % input_img.cols;
            y = regions[i] / input_img.cols;

            seedPoints.push_back(Point(x,y));
        }

        Rect rect1(0,0,0,0);

        regioni_FF.clear();
        regioni_FF.resize(nregions);
        Mat Mask_floodfill = Mask_floodfill.zeros(input_img.rows +2 , input_img.cols +2 ,CV_8UC1);

        for (int i = 0 ; i < nregions; ++i)
        {
            Mask_floodfill = Scalar(0);
            floodFill(input_img,Mask_floodfill, seedPoints[i],Scalar(255),&rect1,
                      0,255 - (int)input_img.at<unsigned char>(seedPoints[i].y,seedPoints[i].x),FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY);

            for (int k = 1; k < Mask_floodfill.rows-1; k++ )
            {
                for (int j = 1; j < Mask_floodfill.cols-1; j++)
                {
                    if ( Mask_floodfill.at<unsigned char>(k,j) == 1 )

                    {   Mask_floodfill.at<unsigned char>(k, j)=255;
                        regioni_FF[i].push_back(Point(j-1,k-1));
                    }
                }
            }
        }
    }

    if (dark_on_bright)
    {   //vl_mser_pix = vl_mser_pix = unsigned 8bit integer
        // load pixel value inside data
        int k=0;
        for(int i=0; i<input_img.rows; i++)
        {
            for (int j=0; j<input_img.cols; j++)
            {
                data[k]=(vl_uint8) (input_img.at<unsigned char>(i,j));
                k++;
            }
        }

        vl_mser_process (filt, (vl_mser_pix*) data) ;

        nregions=0;
        /* Save result  */
        nregions = vl_mser_get_regions_num (filt) ;
        cout << "numero regioni trovate (Dark_On_Bright) : " << nregions << endl;
        regions  = vl_mser_get_regions     (filt) ;

        //conversion from seed_regions to seed_point
        vector < Point > seedPoints;
        seedPoints.clear();
        int x,y;
        for (int i = 0 ; i < nregions ; i++)
        {
            x = regions[i] % input_img.cols;
            y = regions[i] / input_img.cols;
            seedPoints.push_back(Point(x,y));
        }

        Rect rect1(0,0,0,0);

        regioni_FF.clear();
        regioni_FF.resize(nregions);


        Mat Mask_floodfill = Mask_floodfill.zeros(input_img.rows +2 , input_img.cols +2 ,CV_8UC1);

        for (int i = 0 ; i < nregions; ++i)
        {
            Mask_floodfill = Scalar(0);

            floodFill(input_img,Mask_floodfill, seedPoints[i],Scalar(255),&rect1,
                      (int)input_img.at<unsigned char>(seedPoints[i].y,seedPoints[i].x),0,FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY);

            for (int k = 1; k < Mask_floodfill.rows-1; k++ )
            {
                for (int j = 1; j < Mask_floodfill.cols-1; j++)
                {
                    if ( Mask_floodfill.at<unsigned char>(k,j) == 1 )

                    {   Mask_floodfill.at<unsigned char>(k, j)=255;
                        regioni_FF[i].push_back(Point(j-1,k-1));
                    }
                }
            }
        }
    }

    /// release filter
    if (filt)
    {
        vl_mser_delete (filt) ;
        filt = 0 ;
    }

    /// release image data
    if (data)
    {
        free (data) ;
        data = 0 ;
    }
    ///drow the regions
    for(unsigned int j=0; j < regioni_FF.size(); j++)
    {
        for(unsigned int k=0; k < regioni_FF[j].size() ; k++)
        {
            output_img->at<unsigned char>(regioni_FF[j][k].y,regioni_FF[j][k].x) = 255;
        }
    }
    return 0;
}

