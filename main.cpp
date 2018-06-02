#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

string wn1 = "debug threshold";
int threshlow = 10;


void laplacian(Mat src, Mat &sharp, Mat &lapla) {

    // #define show_sharp
    // #define show_lapla 

    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
    sharp = src;

    filter2D(sharp, lapla, CV_32F, kernel);
    
    src.convertTo(sharp, CV_32F);
    
    sharp = sharp - lapla;

    sharp.convertTo(sharp, CV_8UC3);
    lapla.convertTo(lapla, CV_8UC3);
}


Mat edgeotsu(Mat frame) {
    Mat morph = frame.clone();
    for (int r = 1; r < 4; r++)
    {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2*r+1, 2*r+1));
        morphologyEx(morph, morph, CV_MOP_CLOSE, kernel);
        morphologyEx(morph, morph, CV_MOP_OPEN, kernel);
    }
    /* take morphological gradient */
    Mat mgrad;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(morph, mgrad, CV_MOP_GRADIENT, kernel);

    Mat ch[3], merged;
    /* split the gradient image into channels */
    split(mgrad, ch);
    /* apply Otsu threshold to each channel */
    threshold(ch[0], ch[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(ch[1], ch[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(ch[2], ch[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    /* merge the channels */
    merge(ch, 3, merged);

    Mat lapla;
    laplacian(merged, merged, lapla);

    return merged;
}

void blackbg(Mat &src) {
    for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }
}

void distran(Mat in, Mat &out) {
    #define show_distran

    distanceTransform(in, out, CV_DIST_L2, 3);
    normalize(out, out, 0, 1., NORM_MINMAX);
    
    #ifdef show_distran
    imshow("distance transform", out);
    #endif
}

void bin(Mat in, Mat &out) {
    #define show_bin

    cvtColor(in, out, CV_BGR2GRAY);
    threshold(out, out, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    #ifdef show_bin
    imshow("Binary Image", out);
    #endif
}

void peak(Mat in, Mat &out) {
    #define show_peak

    threshold(in, out, .4, 1., CV_THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(out, out, kernel1);

    #ifdef show_peak
    imshow("Peaks", out);
    #endif
}

vector<Vec3b> gencolors(vector<vector<Point> > contours) {
    vector<Vec3b> colors;

    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    return colors;
}

void fillcolor(Mat markers, vector<Vec3b> colors, int size, Mat &out) {
    #define show_fillcolor

    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(size))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    out = dst.clone();

    #ifdef show_fillcolor
    imshow("fill color", out);
    #endif
}

void markercontour(Mat in, Mat &out, vector<vector<Point> > &contours) {
    #define show_markers_contour

    Mat inbk = in.clone();
    in.convertTo(in, CV_8U);

    findContours(in, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    out = Mat::zeros(inbk.size(), CV_32SC1);

    for (size_t i = 0; i < contours.size(); i++)
        drawContours(out, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    circle(out, Point(5,5), 3, CV_RGB(255,255,255), -1);

    #ifdef show_markers_contour
    imshow("markers contour", out*10000);
    #endif
}

void markerwatershed(Mat src, Mat markers, Mat &out) {
    #define show_markers_watershed

    watershed(src, markers);

    out = Mat::zeros(markers.size(), CV_8UC1);
    
    markers.convertTo(out, CV_8UC1);
    bitwise_not(out, out);

    #ifdef show_markers_watershed
    imshow("makers watershed", out);
    #endif
}

int sgc(Mat src) {

    blackbg(src);

    // sharpen
    Mat sharp, lapla;
    laplacian(src, sharp, lapla);
    src = sharp;

    // binary image
    Mat bw;
    bin(src, bw);

    // distance transform
    Mat dist;
    distran(bw, dist);

    // peak
    peak(dist, dist);
   
    // contour
    Mat markers;
    vector<vector<Point> > contours;
    markercontour(dist, markers, contours);

    // watershed
    Mat mark;
    markerwatershed(src, markers, mark);

    // coloring
    vector<Vec3b> colors = gencolors(contours);

    Mat dst;
    fillcolor(markers, colors, contours.size(), dst);

}

void debugger(int, void* x) {
    Mat *frame = (Mat *) x;
    sgc(*frame);
}

enum ConvolutionType {   
/* Return the full convolution, including border */
  CONVOLUTION_FULL, 
  
/* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  
/* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};

void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest) {
	cv::Mat source = img;
	if(CONVOLUTION_FULL == type) {
		source = Mat();
		const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
		copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
	}

	cv::Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
	int borderMode = BORDER_CONSTANT;
	cv::Mat fkernel;
	flip(kernel, fkernel, -1);
	cv::filter2D(source, dest, CV_64F, fkernel, anchor, 0, borderMode);

	if(CONVOLUTION_VALID == type) {
		dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
           .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
	}
}

double sqr(double x) {
	return x * x;
}

bool inRange(double val, double l, double r) {
	return (l <= val && val <= r);
}

void waveletTransform(const cv::Mat& img, cv::Mat& edge, double threshold = 0.15) {
    // https://github.com/fpt-corp/DriverlessCarChallenge_2017-2018/blob/master/example/lane_detection/api_lane_detection.cpp
    
	Mat src = img;
	if (img.type() == 16)
		cvtColor(img, src, CV_BGR2GRAY);
	double pi = M_PI;
	int SIZE = src.rows;
	int SIZE1 = src.cols;
	double m = 1.0;
	double dlt = pow(2.0, m);
	int N = 20;
	double A = -1 / sqrt(2 * pi);//M_PI = acos(-1.0)
	cv::Mat phi_x = cv::Mat(N, N, CV_64F);
	cv::Mat phi_y = cv::Mat(N, N, CV_64F);
	for(int idx = 1; idx <= N; ++idx) {
        for(int idy = 1; idy <= N; ++idy) {
			double x = idx - (N + 1) / 2.0;
			double y = idy - (N + 1) / 2.0;
			double coff = A / sqr(dlt) * exp(-(sqr(x) + sqr(y)) / (2 * sqr(dlt)));
			phi_x.at<double>(idx - 1, idy - 1) = (coff * x);
			phi_y.at<double>(idx - 1, idy - 1) = (coff * y);
		}
    }
		
	normalize(phi_x, phi_x);
	normalize(phi_y, phi_y);
	cv::Mat Gx, Gy;
	conv2(src, phi_x, CONVOLUTION_SAME, Gx);
	conv2(src, phi_y, CONVOLUTION_SAME, Gy);
	cv::Mat Grads = cv::Mat(src.rows, src.cols, CV_64F);
	for(int i = 0; i < Gx.rows; ++i)
		for(int j = 0; j < Gx.cols; ++j) {
			double x = Gx.at<double>(i, j);
			double y = Gy.at<double>(i, j);
			double sqx = sqr(x);
			double sqy = sqr(y);
			Grads.at<double>(i, j) = sqrt(sqx + sqy);
		}
	double mEPS = 100.0 / (1LL << 52);//matlab eps = 2 ^ -52
	cv::Mat angle_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 0; i < SIZE; ++i)
		for(int j = 0; j < SIZE1; ++j) {
			double p = 90;
			if (fabs(Gx.at<double>(i, j)) > mEPS) {
				p = atan(Gy.at<double>(i, j) / Gx.at<double>(i, j)) * 180 / pi;
				if (p < 0) p += 360;
				if (Gx.at<double>(i, j) < 0 && p > 180)
					p -= 180;
				else if (Gx.at<double>(i, j) < 0 && p < 180)
					p += 180;
			}
			angle_array.at<double>(i, j) = p;
		}
	Mat edge_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 1; i < SIZE - 1; ++i) {
        for(int j = 1; j < SIZE1 - 1; ++j) {
			double aval = angle_array.at<double>(i, j);
			double gval = Grads.at<double>(i, j);
			if (inRange(aval,-22.5,22.5) || inRange(aval, 180-22.5,180+22.5)) {
				if (gval > Grads.at<double>(i+1,j) && gval > Grads.at<double>(i-1,j))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if (inRange(aval,90-22.5,90+22.5) || inRange(aval,270-22.5,270+22.5)) {
				if (gval > Grads.at<double>(i, j+1) && gval > Grads.at<double>(i, j-1))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if(inRange(aval,45-22.5,45+22.5) || inRange(aval,225-22.5,225+22.5)) {
				if (gval > Grads.at<double>(i+1,j+1) && gval > Grads.at<double>(i-1,j-1))
					edge_array.at<double>(i,j) = gval;
			}
			else
				if (gval > Grads.at<double>(i+1,j-1) && gval > Grads.at<double>(i-1,j+1))
					edge_array.at<double>(i, j) = gval;
		}
    }
		
	double MAX_E = edge_array.at<double>(0, 0);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j)
			if (MAX_E < edge_array.at<double>(i, j))
				MAX_E = edge_array.at<double>(i, j);
	edge = Mat::zeros(src.rows, src.cols, CV_8U);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j) {
			edge_array.at<double>(i, j) /= MAX_E;
			if (edge_array.at<double>(i, j) > threshold)
				edge.at<uchar>(i, j) = 255;
			else
				edge.at<uchar>(i, j) = 0;
		}
}

Mat sobel(Mat src) {

    Mat src_gray, grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    int c;

    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor( src, src_gray, COLOR_RGB2GRAY );

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return grad;

}

Rect getQuarter(Mat &src, int quarter) {
    int x, y;
    if (quarter == 0) {
        x = 0;
        y = 0;
    } else if (quarter == 1) {
        x = src.cols / 2;
        y = src.rows / 2;
    } else if (quarter == 2) {
        x = 0;
        y = src.rows / 2;
    } else {
        x = src.cols / 2;
        y = 0;
    }
    Rect roi = Rect(x, y, src.cols / 2, src.rows / 2);
    src = src(roi);

    return roi;
}

int main(int, char** argv)
{
    Mat origin;
    String s = "/home/taquy/Desktop/vid/19.avi";
    VideoCapture cap(0);
    // VideoCapture cap(0);
    Mat frame;
    while (true) {
        cap >> frame;

        // cvtColor(frame, frame, COLOR_RGB2BGR);
        origin = frame.clone();

        Mat cannyedge;
        waveletTransform(origin, cannyedge, 0.06);

        imshow("origin", origin);

        Mat thresh;
        cv::threshold(origin, thresh, 100, 255, cv::THRESH_BINARY);

        Mat sobeledge = sobel(origin);

        Mat otsu = edgeotsu(origin);
        // imshow("origin", sobeledge);
        // imshow("cannyedge", cannyedge);
            // imshow("otsu", otsu);

        Mat frame = Mat::zeros(origin.size(), CV_8UC3);


        Rect q1 = getQuarter(thresh, 1);
        Rect q2 = getQuarter(cannyedge, 2);
        Rect q3 = getQuarter(sobeledge, 3);
        Rect q4 = getQuarter(otsu, 0);

        cv::cvtColor(cannyedge, cannyedge, COLOR_GRAY2BGR);
        cv::cvtColor(sobeledge, sobeledge, COLOR_GRAY2BGR);

        imshow("test1", thresh);
        imshow("test2", cannyedge);
        imshow("test3", sobeledge);
        imshow("test4", otsu);

        thresh.copyTo(frame(q1));
        cannyedge.copyTo(frame(q2));
        sobeledge.copyTo(frame(q3));
        otsu.copyTo(frame(q4));

        imshow("frame", frame);
        

        // Mat m3;
        // cv::hconcat(q1, q2, m3);
        // assert(10 == m3.rows && 8 == m3.cols);

        // imshow("m3", m3);

        char k = waitKey(1);
        if (k == 113) break;
        if (k == 32) waitKey(); 
        if (k == 115) {
            imwrite("frame.jpg", frame);
        }
    }
    cap.release();
    cvDestroyAllWindows();
    return 0;
}