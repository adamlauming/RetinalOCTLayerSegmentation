/*==============================================================================================
Program: Segmentation
Author:  Ming Liu (lauadam0730@gmail.com)
Version: 1.0
Data:    2020/12/19
Copyright(c): MIPAV Lab (mipav.net), Soochow University & Bigvision Company(bigvisiontech.com).
              2020-Now. All rights reserved.
See LICENSE.txt for details
===============================================================================================*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include "SGSmooth.hpp"

using namespace std;
using namespace cv;

void findRPE(const cv::Mat& inImg, vector<int>& RPE){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    // parameters
    int N = row/2 - 50; // vertical ranges
    int p = 1;   // threshold of jump pixels
    double lamda = 50; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (fltImg.at<uchar>(i+N+row/2-N, 1) + fltImg.at<uchar>(i+N+k+row/2-N, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - fltImg.at<uchar>(i+N+row/2-N, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    RPE[column-1] = i_min[column-1] - 1 + row/2 - N;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    RPE[column-2] = i_min[column-2] - N - 1 + row/2;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        RPE[n-1] = i_min[n-1] - N - 1 + row/2;
    }
}

void findILM(const cv::Mat& inImg, vector<int>& RPE, vector<int>& ILM){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 11);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    for (int i = 0; i < YGrad.rows; i++) {
        char * p = YGrad.ptr<char>(i);
        for (int j = 0; j < YGrad.cols; j++) {
            if (p[j] < 0) p[j] = 0;
        }
    }
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = RPE[j]-30; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS2", YGradientABS);
//    waitKey();

    int RPELoc = *max_element(RPE.begin(),RPE.end());
    // parameters
    int N = (RPELoc - 30 - 50)/2; // vertical ranges: 2*N+1
    int p = 1;   // threshold of jump pixels
    double lamda = 1; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    ILM[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    ILM[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        ILM[n-1] = i_min[n-1] - 1 + 50;
    }
}

void findONL(const cv::Mat& inImg, vector<int>& RPE, vector<int>& ONL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 7);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = RPE[j]-8; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = 0; i < RPE[j] - 30; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS", YGradientABS);
//    waitKey();

    int RPELoc = *max_element(RPE.begin(),RPE.end());
    // parameters
    int N = (RPELoc - 50)/2; // vertical ranges: 2*N+1
    int p = 3;   // threshold of jump pixels
    double lamda = 10; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    ONL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    ONL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        ONL[n-1] = i_min[n-1] - 1 + 50;
    }
}

void findOPL(const cv::Mat& inImg, vector<int>& ONL, int& foveaLoc, vector<int>& OPL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 11);
    Mat YGrad, YNegGradABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    for (int i = 0; i < YGrad.rows; i++) {
        short * p = YGrad.ptr<short>(i);
        for (int j = 0; j < YGrad.cols; j++) {
            if (p[j] > 0) p[j] = 0;
        }
    }
    convertScaleAbs(YGrad, YNegGradABS);
    for (int j = 0; j < foveaLoc-150; j++){
        for (int i = 0; i < ONL[j]-30; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc-150; j < foveaLoc+150; j++){
        for (int i = 0; i < ONL[j]-40; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc+150; j < column; j++){
        for (int i = 0; i < ONL[j]-30; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < column; j++){
        for (int i = ONL[j]-10; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YNegGradABS.png", YNegGradABS);
//    imshow("YNegGradABS", YNegGradABS);
//    waitKey();
    int ONLLoc = *max_element(ONL.begin(),ONL.end());
    // parameters
    int N = (ONLLoc - 50)/2; // vertical ranges: 2*N+1
    int p = 1;   // threshold of jump pixels
    double lamda = 10; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YNegGradABS.at<uchar>(i+N+50, 1) + YNegGradABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YNegGradABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    OPL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    OPL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        OPL[n-1] = i_min[n-1] - 1 + 50;
    }

    for (int n = 0; n < column; n++){
        if ((ONL[n] - OPL[n]) < 15) OPL[n] = ONL[n] - 15;
    }
}

void findINL(const cv::Mat& inImg, vector<int>& OPL, int& foveaLoc, vector<int>& INL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    Mat YGrad, YGradientABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    convertScaleAbs(YGrad, YGradientABS);
    for (int j = 0; j < column; j++){
        for (int i = 0; i < OPL[j]-10; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < foveaLoc-7; j++){
        for (int i = OPL[j]-2; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc-7; j < foveaLoc+7; j++){
        for (int i = OPL[j]+1; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc+7; j < column; j++){
        for (int i = OPL[j]-2; i < row; i++){
            YGradientABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YGradientABS.png", YGradientABS);
//    imshow("YGradientABS", YGradientABS);
//    waitKey();
    int OPLLoc = *max_element(OPL.begin(),OPL.end());
    // parameters
    int N = (OPLLoc - 50)/2; // vertical ranges: 2*N+1
    int p = 1;   // threshold of jump pixels
    double lamda = 1; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YGradientABS.at<uchar>(i+N+50, 1) + YGradientABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YGradientABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    INL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    INL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        INL[n-1] = i_min[n-1] - 1 + 50;
    }

    for (int n = 0; n < column; n++){
        if ((OPL[n] - INL[n]) < 1) INL[n] = OPL[n] - 1;
    }
}

void findIPL(const cv::Mat& inImg, vector<int>& INL, int& foveaLoc, vector<int>& IPL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    Mat YGrad, YNegGradABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    for (int i = 0; i < YGrad.rows; i++) {
        short * p = YGrad.ptr<short>(i);
        for (int j = 0; j < YGrad.cols; j++) {
            if (p[j] > 0) p[j] = 0;
        }
    }
    convertScaleAbs(YGrad, YNegGradABS);
    for (int j = 0; j < column; j++){
        for (int i = 0; i < INL[j]-10; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = 0; j < foveaLoc-7; j++){
        for (int i = INL[j]-1; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc-7; j < foveaLoc+7; j++){
        for (int i = INL[j]+2; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc+7; j < column; j++){
        for (int i = INL[j]-1; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YNegGradABS.png", YNegGradABS);
//    imshow("YNegGradABS", YNegGradABS);
//    waitKey();

    int INLLoc = *max_element(INL.begin(),INL.end());
    // parameters
    int N = (INLLoc - 50)/2; // vertical ranges: 2*N+1
    int p = 1;   // threshold of jump pixels
    double lamda = 1; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YNegGradABS.at<uchar>(i+N+50, 1) + YNegGradABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YNegGradABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    IPL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    IPL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        IPL[n-1] = i_min[n-1] - 1 + 50;
    }

    for (int n = 0; n < foveaLoc-7; n++){
        if ((INL[n] - IPL[n]) < 1) IPL[n] = INL[n] - 1;
    }
    for (int n = foveaLoc-7; n < foveaLoc+7; n++){
        if ((INL[n] - IPL[n]) < 0) IPL[n] = INL[n];
    }
    for (int n = foveaLoc+7; n < column; n++){
        if ((INL[n] - IPL[n]) < 1) IPL[n] = INL[n] - 1;
    }
}

void findNFL(const cv::Mat& inImg, vector<int>& IPL, vector<int>& ILM, int& foveaLoc, vector<int>& NFL){
    int row = inImg.rows;
    int column = inImg.cols;
    Mat fltImg;
    medianBlur(inImg, fltImg, 3);
    Mat YGrad, YNegGradABS;
    Sobel(fltImg, YGrad, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT );
    for (int i = 0; i < YGrad.rows; i++) {
        short * p = YGrad.ptr<short>(i);
        for (int j = 0; j < YGrad.cols; j++) {
            if (p[j] > 0) p[j] = 0;
        }
    }
    convertScaleAbs(YGrad, YNegGradABS);
    for (int j = 0; j < foveaLoc-7; j++){
        for (int i = IPL[j]-10; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc-7; j < foveaLoc+7; j++){
        for (int i = IPL[j]+10; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
    for (int j = foveaLoc+7; j < column; j++){
        for (int i = IPL[j]-10; i < row; i++){
            YNegGradABS.at<uchar>(i,j) = 0;
        }
    }
//    imwrite("YNegGradABS.png", YNegGradABS);
//    imshow("YNegGradABS", YNegGradABS);
//    waitKey();

    int IPLLoc = *max_element(IPL.begin(),IPL.end());
    // parameters
    int N = (IPLLoc - 50)/2; // vertical ranges: 2*N+1
    int p = 1;   // threshold of jump pixels
    double lamda = 1; // Regularisation parameter

    // calculate the dynamic programming equation kept in d[][][]
    vector<vector<vector<double> > > d((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column)));
    // the initialization equation: n = 1;
    for (int i = -N; i <= N; i++){
        for (int k = -p; k <= p; k++){
            d[i+N][k+p][1] = - (YNegGradABS.at<uchar>(i+N+50, 1) + YNegGradABS.at<uchar>(i+N+k+50, 0));
        }
    }

    // the recursion formula for the dynamic programming routine
    vector<vector<vector<double> > > cost((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the cost kept in cost[][][]
    vector<vector<vector<double> > > cost_min((2*N+1), vector<vector<double> >( (2*p+1), vector<double>(column))); // calculate the minimum cost of cost[i+k+N][][n] kept in cost_min
    vector<vector<vector<int> > > l_min((2*N+1), vector<vector<int> >( (2*p+1), vector<int>(column))); // when l is minimum, the cost_min is achieved and it's kept in l_min
    for (int n = 2; n < column; n++){
        for (int i = -N; i <= N; i++){
            for (int k = -p; k <= p; k++){
                if ((i+k+N)>=0 && ((i+k+N)<=(2*N))){ // limit the boundry of d[][][]
                    for (int l = -p; l <= p; l++){
                        cost[i+k+N][l+p][n] = d[i+k+N][l+p][n-1] + lamda * (l-k) * (l-k);
                    }
                    vector<double> vec_temp(2*p+1);
                    for (int u = 0; u < (2*p+1); u++){
                        vec_temp[u] = cost[i+k+N][u][n];
                    }
                    cost_min[i+k+N][k+p][n] = *min_element(vec_temp.begin(),vec_temp.end());
                    l_min[i+k+N][k+p][n] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;  /// Notice!!!  index+1
                    d[i+N][k+p][n] = - YNegGradABS.at<uchar>(i+N+50, n) + cost_min[i+k+N][k+p][n];
                }
            }
        }
    }

    // backtracking
    vector <double> dn_min(2*p+1); // keep the minimum of the last d[][k][column-1] in the range of drift (2*N+1),every k has a value.
    vector <double> ii_min(2*p+1); // keep the position value of the minimum d[ii][k][column-1]
    for (int k = 0; k < (2*p+1); k++){
        vector<double> vec_temp(2*N+1);
        for (int u = 0; u < (2*N+1); u++){
            vec_temp[u] = d[u][k][column-1];
        }
        dn_min[k] = *min_element(vec_temp.begin(),vec_temp.end());
        ii_min[k] = min_element(vec_temp.begin(),vec_temp.end()) - vec_temp.begin() + 1;
    }
//    int d_min = *min_element(dn_min.begin(),dn_min.end());
    int k_min = min_element(dn_min.begin(),dn_min.end()) - dn_min.begin() + 1;
    vector <double> i_min(column);
    i_min[column-1] = ii_min[k_min-1];
    NFL[column-1] = i_min[column-1] - 1 + 50;
    vector <int> km(column);
    km[column-1] = k_min - p - 1;
    i_min[column-2] = i_min[column-1] + km[column-1];
    NFL[column-2] = i_min[column-2] - 1 + 50;
    for (int n = column -2; n > 0; n--){
        if(i_min[n+1] < 2) i_min[n+1] = 2;
        if(i_min[n+1] > 2*N) i_min[n+1] = 2*N; // keep the boundry
        km[n] = l_min[i_min[n+1]-1][km[n+1]+p][n+1] - p - 1;
        i_min[n-1] = i_min[n] + km[n];
        NFL[n-1] = i_min[n-1] - 1 + 50;
    }

    for (int n = 0; n < column; n++){
        if ((IPL[n] - NFL[n]) < 1) NFL[n] = IPL[n] - 1;
//        if ((NFL[n] - ILM[n]) < 0) NFL[n] = ILM[n];
    }

    // judge which side has the optic disk.
    int diffLeft = 0;
    int diffRight = 0;
    for (int n = foveaLoc-200; n < foveaLoc; n++){
        int tempDiff = NFL[n] - ILM[n];
        diffLeft += tempDiff;
    }
    for (int n = foveaLoc+1; n <= foveaLoc+200; n++){
        int tempDiff = NFL[n] - ILM[n];
        diffRight += tempDiff;
    }
    // cout << diffLeft << ",," << diffRight << endl;
    if (diffLeft > diffRight){ // the optic disk is on the left.
        for (int n = foveaLoc; n < column; n++){
            if ((NFL[n] - ILM[n]) > 5) NFL[n] = ILM[n] + 5;
        }
    }
    else { // the optic disk is on the right.
        for (int n = 0; n <= foveaLoc; n++){
            if ((NFL[n] - ILM[n]) > 5) NFL[n] = ILM[n] + 5;
        }
    }

    for (int n = foveaLoc-10; n < foveaLoc+10; n++){
        if ((NFL[n] - ILM[n]) > 5) NFL[n] = ILM[n] + 5;
    }
}

//int main() {
//    Mat image = imread("Bscan4.png", IMREAD_GRAYSCALE);
//    //Mat image = imread("pathology_enhance2.png", IMREAD_GRAYSCALE);
//    Mat seg;
////    int row = image.rows;
//    int column = image.cols;
//
//    vector <int> layerRPE(column);
//    findRPE(image, layerRPE);
//    vector <int> layerILM(column);
//    findILM(image, layerRPE, layerILM);
//    vector <int> layerONL(column);
//    findONL(image, layerRPE, layerONL);
//    vector <int> foveadiff(200);
//    for (int n = column/2-100; n < column/2+100; n++){
//        foveadiff[n-(column/2-100)] = layerONL[n] - layerILM[n];
//    }
//    int diffLoc = min_element(foveadiff.begin(),foveadiff.end()) - foveadiff.begin() + 1;
//    int foveaLoc = column/2 - 100 + diffLoc;
//    vector <int> layerOPL(column);
//    findOPL(image, layerONL, foveaLoc, layerOPL);
//    vector <int> layerINL(column);
//    findINL(image, layerOPL, foveaLoc, layerINL);
//    vector <int> layerIPL(column);
//    findIPL(image, layerINL, foveaLoc, layerIPL);
//    vector<int> layerNFL(column);
//    findNFL(image, layerIPL, layerILM, foveaLoc, layerNFL);
//
//    for (int n = 0; n < column; ++n) {
//        if ((layerNFL[n] - layerILM[n]) < 0) layerNFL[n] = layerILM[n];
//        if ((layerIPL[n] - layerNFL[n]) < 0) layerIPL[n] = layerNFL[n];
//        if ((layerINL[n] - layerIPL[n]) < 0) layerINL[n] = layerIPL[n];
//        if ((layerOPL[n] - layerINL[n]) < 0) layerOPL[n] = layerINL[n];
//        if ((layerONL[n] - layerOPL[n]) < 0) layerONL[n] = layerOPL[n];
//        if ((layerRPE[n] - layerONL[n]) < 0) layerRPE[n] = layerONL[n];
//    }
//    vector<double> layerILM_double(layerILM.begin(), layerILM.end());
//    vector<double> layerNFL_double(layerNFL.begin(), layerNFL.end());
//    vector<double> layerIPL_double(layerIPL.begin(), layerIPL.end());
//    vector<double> layerINL_double(layerINL.begin(), layerINL.end());
//    vector<double> layerOPL_double(layerOPL.begin(), layerOPL.end());
//    vector<double> layerONL_double(layerONL.begin(), layerONL.end());
//    vector<double> layerRPE_double(layerRPE.begin(), layerRPE.end());
//
//    layerNFL_double = sg_smooth(layerNFL_double, 5, 0);
//    layerILM_double = sg_smooth(layerILM_double, 5, 0);
//    layerIPL_double = sg_smooth(layerIPL_double, 5, 0);
//    layerINL_double = sg_smooth(layerINL_double, 5, 0);
//    layerOPL_double = sg_smooth(layerOPL_double, 5, 0);
//    layerONL_double = sg_smooth(layerONL_double, 5, 0);
//    layerRPE_double = sg_smooth(layerRPE_double, 5, 0);
//
//    cvtColor(image, image, COLOR_GRAY2BGR);
//    for(int n = 0; n < image.cols; n++) {
//        Point ilm(n, layerILM_double[n]);
//        circle(image, ilm, 1, Scalar(0, 0, 255), -1); //红色
//        Point nfl(n, layerNFL_double[n]);
//        circle(image, nfl, 1, Scalar(0, 255, 0), -1); //绿色
////        Point ipl(n, layerIPL_double[n]);
////        circle(image, ipl, 1, Scalar(0, 69, 255), -1); //橘色
////        Point inl(n, layerINL_double[n]);
////        circle(image, inl, 1, Scalar(63, 133, 205), -1); //棕色
//        Point opl(n, layerOPL_double[n]);
//        circle(image, opl, 1, Scalar(175, 39, 5), -1); //深蓝
//        Point onl(n, layerONL_double[n]);
//        circle(image, onl, 1, Scalar(255, 245, 0), -1); //淡蓝
//        Point rpe(n, layerRPE_double[n]);
//        circle(image, rpe, 1, Scalar(0, 255, 255), -1); //黄色
//    }
//    imshow("segmentation", image);
//    waitKey();
//    imwrite("segmentated.jpg", image);
//
//    return 0;
//}


void segmentThreeLayers(cv::Mat inImg, vector<int>& layerRPE, vector<int>& layerILM, vector<int>& layerONL){
    findRPE(inImg, layerRPE);
    vector<double> layerRPE_double(layerRPE.begin(), layerRPE.end());
    layerRPE_double = sg_smooth(layerRPE_double, 5, 0);
    vector<int> layerRPE_int(layerRPE_double.begin(), layerRPE_double.end());
    layerRPE = layerRPE_int;
    findILM(inImg, layerRPE, layerILM);
    vector<double> layerILM_double(layerILM.begin(), layerILM.end());
    layerILM_double = sg_smooth(layerILM_double, 5, 0);
    vector<int> layerILM_int(layerILM_double.begin(), layerILM_double.end());
    layerILM = layerILM_int;
    findONL(inImg, layerRPE, layerONL);
    vector<double> layerONL_double(layerONL.begin(), layerONL.end());
    layerONL_double = sg_smooth(layerONL_double, 5, 0);
    vector<int> layerONL_int(layerONL_double.begin(), layerONL_double.end());
    layerONL = layerONL_int;
}

void calculate(cv::Mat& inImg, cv::Mat& baseImg, cv::Mat& label) {
    vector <int> layerRPE_labels(label.cols);
    vector <int> layerILM_labels(label.cols);
    vector <int> layerONL_labels(label.cols);
//    for (int i = 0; i < label.rows; i++) {
//        cout << "i: " << int(label.at<uchar>(i, 250)) << endl;
//    }
    for (int j = 0; j < label.cols; j++){
        for (int i = 0; i < label.rows; i++) {
            if (label.at<uchar>(i, j) > 100){
                layerILM_labels[j] = i-1 ;
                break;
            }
        }
    }
    for (int j = 0; j < label.cols; j++){
        for (int i = 0; i < label.rows; i++) {
            if (label.at<uchar>(i, j) > 83 && label.at<uchar>(i, j) < 88){
                layerONL_labels[j] = i-4;
                break;
            }
        }
    }

    vector <int> layerRPE(inImg.cols);
    vector <int> layerILM(inImg.cols);
    vector <int> layerONL(inImg.cols);
    segmentThreeLayers(inImg, layerRPE, layerILM, layerONL);

    double errorPixels = 0;
    for (int i = 0; i < layerONL_labels.size(); i++){
//        cout << layerONL_labels[i] << ", " << layerONL[i] << endl;
        errorPixels += abs(layerONL_labels[i] -  layerONL[i]);
    }
    errorPixels = errorPixels / inImg.cols;
    cout << "denoised IS/OS error: " << errorPixels << endl;
    errorPixels = 0;
    for (int i = 0; i < layerILM_labels.size(); i++){
        errorPixels += abs(layerILM_labels[i] -  layerILM[i]);
    }
    errorPixels = errorPixels / inImg.cols;
    cout << "denoised ILM error: " << errorPixels << endl;

    Mat image;
    cvtColor(inImg, image, COLOR_GRAY2BGR);
    for (int n = 0; n < inImg.cols; n++) {
        Point ilm(n, layerILM[n]);
        circle(image, ilm, 1, Scalar(255, 245, 0), -1); //淡蓝
        Point ilm_label(n, layerILM_labels[n]);
        circle(image, ilm_label, 1, Scalar(0, 255, 255), -1); //黄色
        Point onl(n, layerONL[n]);
        circle(image, onl, 1, Scalar(0, 0, 255), -1); //红色
        Point onl_label(n, layerONL_labels[n]);
        circle(image, onl_label, 1, Scalar(0, 255, 0), -1); //绿色
    }
    imwrite("segmentated.jpg", image);

    vector <int> layerRPE_base(baseImg.cols);
    vector <int> layerILM_base(baseImg.cols);
    vector <int> layerONL_base(baseImg.cols);
    segmentThreeLayers(baseImg, layerRPE_base, layerILM_base, layerONL_base);

    errorPixels = 0;
    for (int i = 0; i < layerONL_labels.size(); i++){
        errorPixels += abs(layerONL_labels[i] -  layerONL_base[i]);
    }
    errorPixels = errorPixels / inImg.cols;
    cout << "base IS/OS error: " << errorPixels << endl;
    errorPixels = 0;
    for (int i = 0; i < layerILM_labels.size(); i++){
        errorPixels += abs(layerILM_labels[i] -  layerILM_base[i]);
    }
    errorPixels = errorPixels / inImg.cols;
    cout << "base ILM error: " << errorPixels << endl;

    Mat image_base;
    cvtColor(baseImg, image_base, COLOR_GRAY2BGR);
    for (int n = 0; n < inImg.cols; n++) {
        Point ilm(n, layerILM_base[n]);
        circle(image_base, ilm, 1, Scalar(255, 245, 0), -1); //淡蓝
        Point ilm_label(n, layerILM_labels[n]);
        circle(image_base, ilm_label, 1, Scalar(0, 255, 255), -1); //黄色
        Point onl(n, layerONL_base[n]);
        circle(image_base, onl, 1, Scalar(0, 0, 255), -1); //红色
        Point onl_label(n, layerONL_labels[n]);
        circle(image_base, onl_label, 1, Scalar(0, 255, 0), -1); //绿色
//        Point rpe(n, layerRPE_base[n]);
//        circle(image_base, rpe, 1, Scalar(0, 255, 255), -1); //黄色
    }
    imwrite("segmentated_base.jpg", image_base);
}

int main() {
    Mat Bscan = imread("4_bscan_result.png", 0);
    Mat baseImg = imread("bscan_6.jpg", 0);
    Mat label = imread("henanshengliyankeyiyuan_20201211_154301_label_6.jpg", 0);
    calculate(Bscan, baseImg, label);
}

//void CalculateEPI(cv::Mat& inImg, cv::Mat& baseImg) {
//    vector <int> layerRPE(inImg.cols);
//    findRPE(inImg, layerRPE);
//    vector<double> layerRPE_double(layerRPE.begin(), layerRPE.end());
//    layerRPE_double = sg_smooth(layerRPE_double, 5, 0);
//    vector<int> layerRPE_int(layerRPE_double.begin(), layerRPE_double.end());
//    vector <int> layerILM(inImg.cols);
//    findILM(inImg, layerRPE, layerILM);
//    vector<double> layerILM_double(layerILM.begin(), layerILM.end());
//    layerILM_double = sg_smooth(layerILM_double, 5, 0);
//    vector<int> layerILM_int(layerILM_double.begin(), layerILM_double.end());
//    vector <int> layerONL(inImg.cols);
//    findONL(inImg, layerRPE, layerONL);
//    vector<double> layerONL_double(layerONL.begin(), layerONL.end());
//    layerONL_double = sg_smooth(layerONL_double, 5, 0);
//    vector<int> layerONL_int(layerONL_double.begin(), layerONL_double.end());
//
//    int sum_rpe = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_rpe += abs(inImg.at<uchar>(i + layerRPE_int[j] + 1, j) - inImg.at<uchar>(i + layerRPE_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_rpe += abs(inImg.at<uchar>(i + layerRPE_int[j] + 1, j) - inImg.at<uchar>(i + layerRPE_int[j], j));
//        }
//    }
//    int sum_ilm = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_ilm += abs(inImg.at<uchar>(i + layerILM_int[j] + 1, j) - inImg.at<uchar>(i + layerILM_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_ilm += abs(inImg.at<uchar>(i + layerILM_int[j] + 1, j) - inImg.at<uchar>(i + layerILM_int[j], j));
//        }
//    }
//    int sum_onl = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_onl += abs(inImg.at<uchar>(i + layerONL_int[j] + 1, j) - inImg.at<uchar>(i + layerONL_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_onl += abs(inImg.at<uchar>(i + layerONL_int[j] + 1, j) - inImg.at<uchar>(i + layerONL_int[j], j));
//        }
//    }
//    double sum = sum_rpe + sum_ilm + sum_onl;
//
//    int sum_rpe_b = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_rpe_b += abs(baseImg.at<uchar>(i + layerRPE_int[j] + 1, j) - baseImg.at<uchar>(i + layerRPE_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_rpe_b += abs(baseImg.at<uchar>(i + layerRPE_int[j] + 1, j) - baseImg.at<uchar>(i + layerRPE_int[j], j));
//        }
//    }
//    int sum_ilm_b = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_ilm_b += abs(baseImg.at<uchar>(i + layerILM_int[j] + 1, j) - baseImg.at<uchar>(i + layerILM_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_ilm_b += abs(baseImg.at<uchar>(i + layerILM_int[j] + 1, j) - baseImg.at<uchar>(i + layerILM_int[j], j));
//        }
//    }
//    int sum_onl_b = 0;
//    for (int i = -3; i <= 3; i++) {
//        for (int j = 10; j < 680; j++) {
//            sum_onl_b += abs(baseImg.at<uchar>(i + layerONL_int[j] + 1, j) - baseImg.at<uchar>(i + layerONL_int[j], j));
//        }
//        for (int j = 850; j < 950; j++) {
//            sum_onl_b += abs(baseImg.at<uchar>(i + layerONL_int[j] + 1, j) - baseImg.at<uchar>(i + layerONL_int[j], j));
//        }
//    }
//    double sum_b = sum_rpe_b + sum_ilm_b + sum_onl_b;
//    double EPI = sum / sum_b;
//    cout << "The EPI of this image: " << EPI << endl;
//    double EPI_2 = sum_b / sum;
//    cout << "The EPI of this image: " << EPI_2 << endl;
//
//    Mat image;
//    cvtColor(baseImg, image, COLOR_GRAY2BGR);
//    for (int n = 0; n < 1000; n++) {
//        Point ilm(n, layerILM_double[n]);
//        circle(image, ilm, 1, Scalar(255, 245, 0), -1); //淡蓝
//        Point onl(n, layerONL_double[n]);
//        circle(image, onl, 1, Scalar(0, 0, 255), -1); //淡蓝
//        Point rpe(n, layerRPE_double[n]);
//        circle(image, rpe, 1, Scalar(0, 255, 255), -1); //黄色
//    }
//    //for (int n = 850; n < 940; n++) {
//    //	Point ilm(n, layerILM_double[n]);
//    //	circle(image, ilm, 1, Scalar(255, 245, 0), -1); //淡蓝
//    //	Point onl(n, layerONL_double[n]);
//    //	circle(image, onl, 1, Scalar(0, 0, 255), -1); //淡蓝
//    //	Point rpe(n, layerRPE_double[n]);
//    //	circle(image, rpe, 1, Scalar(0, 255, 255), -1); //黄色
//    //}
//    //imshow("segmentation", image);
//    //waitKey();
//    imwrite("segmentated.jpg", image);
//}
//
//int main() {
//    Mat Bscan = imread("image_base3.png", 0);
//    Mat baseImg = imread("image_strollr_3.png", 0);
//    CalculateEPI(Bscan, baseImg);
//}
