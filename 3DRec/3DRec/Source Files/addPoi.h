#include <iostream>
#include <string>
#include <iomanip>  
#include <sstream> 

#include "opencv/cv.h"

#include "dataTypeStructs.h"

using namespace std;
using namespace cv;

#define DEBUG 0


class addPoi {

private:
	Mat Points2d;
	Mat Points3d;

	Mat overlapping;

public:
	void addingInitialPoints(vector<KeyPoint> keypoints, vector<SpacePoint> pointCloud, Mat calibrationMatrix);

	addPoi();

	Mat get2dPoints();

	Mat get3dPoints();

	Mat lookup3D(KeyPoint keypoint);

	bool hasKeypoint(KeyPoint keypoint);

	Mat getCameraMatrix();

	void setUpNewImage(vector<KeyPoint> newKeyPoints1, Vector<KeyPoint> newKeyPoints2);

};