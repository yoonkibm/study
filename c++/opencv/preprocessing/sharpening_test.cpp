#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void UnSharpen(const Mat& src, Mat& dst, Size ksize, double sigma, float alpha){
    Mat gblur;
    GaussianBlur(src, gblur, ksize, sigma, sigma);

    dst = src + alpha*(src - gblur);
}

void LaplacianSharpen(const Mat& src, Mat& dst, int direction, float alpha){
    Mat laplaicianFilter;
    Mat kern;

    if(direction == 4){
        kern = (Mat_<double>(3, 3) << 
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0);
    }
    else if(direction == 8){
        kern = (Mat_<double>(3, 3) << 
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1);
    }
    else{
        cerr << "The direction variable was entered incorrectly. Please enter either 4 or 8." << endl;
    }
    
    filter2D(src, laplaicianFilter, src.depth(), kern, Point(-1, -1));

    dst = src + alpha * laplaicianFilter;
}

int main(){

    Mat lena_img = imread("lena.jpg", IMREAD_COLOR);
    if(lena_img.empty()){
        cerr << "Can't call image, Check your directory." << endl;
        return -1;
    }

    Mat img;
    cvtColor(lena_img, img, COLOR_BGR2GRAY);

    Size size1(3, 3);
    double sigma = 4.0;

    Mat gblur;
    GaussianBlur(img, gblur, size1, sigma, sigma);

    Mat sharpening_dst;
    UnSharpen(img, sharpening_dst, size1, sigma, 1.0);

    Mat lapsharp_dst;
    LaplacianSharpen(img, lapsharp_dst, 8, 0.5);

    imshow("Source Image", img);
    imshow("Unsharpening Image", sharpening_dst);
    imshow("Laplacian Sharpening Image", lapsharp_dst);

    waitKey(0);

    return 0;
}