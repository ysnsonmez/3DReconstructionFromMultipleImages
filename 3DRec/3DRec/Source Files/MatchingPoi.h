#include <iostream>
#include <string> 
#include <iomanip>
#include <sstream> 

#include "opencv/cv.h"

using namespace std;
using namespace cv;

struct Colours {
	int red;
	int blue;
	int green;
};

class MatchingPoints {
private:
	vector<KeyPoint> keyPoints1;
	vector<KeyPoint> keyPoints2;
	vector<Colours> pointColours;
	Mat fundamentalMatrics;

	vector<DMatch> matches;
	vector<KeyPoint> fullKeypoints1, fullKeypoints2;
	bool enoughMatches;
public:
	MatchingPoints(Mat image1, Mat image2);
	vector<KeyPoint> getKeyPoints1();
	vector<KeyPoint> getKeyPoints2();
	Mat getFundamentalMatrix();
	void displayFull(Mat image1, Mat image2);
	bool hasEnoughMatches();
	vector<Colours> getColours(Mat frame1);
};
