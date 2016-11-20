#include <iostream> 
#include <string>   
#include <iomanip> 
#include <sstream> 

#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/core/core.hpp>      
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv/cv.h"
#include "matcher.h"
#include "MatchingPoi.h"
#include "MatrixCalc.h"

using namespace std;
using namespace cv;

MatchingPoints::MatchingPoints (Mat image1, Mat image2) {
	Matcher goodMatcher;
	goodMatcher.setConfidenceLevel(0.98);
	goodMatcher.setMinDistanceToEpipolar(1.0);
	goodMatcher.setRatio(0.65f);
	cv::Ptr<cv::FeatureDetector> pfd = new cv::SurfFeatureDetector(10);
	goodMatcher.setFeatureDetector(pfd);		

	// Match the two images
	enoughMatches = goodMatcher.match(image1,image2,
		matches, fullKeypoints1, fullKeypoints2);

	fundamentalMatrics = goodMatcher.getFundamentalMatrix();

	//DMatches contains index for which two keypoint vectors that matches well
	for (int i = 0; i < matches.size(); i++) {
		DMatch match = matches.at(i);
		keyPoints1.push_back(fullKeypoints1.at(match.queryIdx));
		keyPoints2.push_back(fullKeypoints2.at(match.trainIdx));
	}
}

vector<Colours> MatchingPoints::getColours(Mat image) {
	cout << "Image dimentions " << image.cols << "," << image.rows << endl;

	vector<Colours> colours;
	KeyPoint point;



	for (int i = 0; i < keyPoints1.size(); i++) {	
		
		point = keyPoints1.at(i);
		int x = int (point.pt.x + 0.5);
		int y = int (point.pt.y + 0.5);
		if (DEBUG == 1) {
			cout << "Coordinates are (" << x << "," << y << ")" << endl ;
		}
		Point3_<uchar> *p = image.ptr< Point3_<uchar>> (y,x);		


		Colours pointColour;
		
		
		pointColour.blue = int(p->x);
		pointColour.green = int(p->y);		
		pointColour.red = int(p->z);


		

		colours.push_back(pointColour);
		if (DEBUG == 1) {
			cout << "BGR values are (" << pointColour.blue <<"," << pointColour.green << "," << pointColour.red << ")" << endl << endl;
		}
	}
	pointColours = colours;

	return colours;
}

Mat MatchingPoints::getFundamentalMatrix() 
{
	return fundamentalMatrics;
}

vector<KeyPoint> MatchingPoints::getKeyPoints1() 
{
	return keyPoints1;
}

vector<KeyPoint> MatchingPoints::getKeyPoints2() 
{
	return keyPoints2;
}

void MatchingPoints::displayFull(Mat image1, Mat image2) 
{
	Mat imageMatches;
	drawMatches(image1, fullKeypoints1, image2, fullKeypoints2, matches, imageMatches);
	namedWindow("Matches", WINDOW_NORMAL);
	moveWindow("Matches", 220, 400);
	imshow("Matches", imageMatches);
	imwrite("Output.jpg", imageMatches);
}

bool MatchingPoints::hasEnoughMatches() 
{
	return enoughMatches;
}