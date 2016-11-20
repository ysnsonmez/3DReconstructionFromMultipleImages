#include <iostream> 
#include <string>   
#include <iomanip>  
#include <sstream>  
#include <math.h>

#include <opencv2\imgproc\imgproc.hpp>  
#include <opencv2\core\core.hpp>        
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv\cv.h>

#include "dataTypeStructs.h"
#include "matcher.h"
#include "MatchingPoi.h"
#include "MatrixCalc.h"
#include "buildModel.h"
#include "PCT.h"

using namespace std;
using namespace cv;

void nearest(Mat object, Mat image);
void downsample(Mat *image);
vector<KeyPoint> *robustMatching(Mat image1, Mat image2);
Matx34d tableProcess(Matx34d P1,
	vector<KeyPoint> newKeyPoints,
	vector<KeyPoint> oldKeyPoints,
	PCTable *current,
	PCTable *previous, Mat K);

int fileNumber;

int main(int argc, char *argv[], char *window_name) {

	// 3Drec rootname imagecount extension -> 3Drec im 15 .jpg
	//if (argc != 4) 
	//{
	//	cout << "Not enough parameters" << endl;
	//	return -1;
	//}

	fileNumber = 0;
	argv[1] = "im";
	argv[2] = "15";
	argv[3] = ".jpg";

	const string image = argv[1];
	const string arg2 = argv[2];
	const string extension = argv[3];
	int numberPictures = atoi(arg2.c_str());

	int pictureNumber1 = 0;
	int pictureNumber2 = 1;


	string imageName1 = image + "0" + extension;
	string imageName2 = image + "1" + extension;


	Mat frame1 = imread(imageName1, -1);
	Mat frame2 = imread(imageName2, -1);


	MatrixCalc matrixCalculator;
	buildModel buildModel;

	vector<SpacePoint> pointCloud;
	Matx34d P;
	Matx34d P1;

	int previousNumberOfPointsAdded = 0;

	bool initial3dModel = true;

	PCTable table1;
	PCTable table2;

	table2.init();
	table1.init();

	PCTable *current = &table1;
	PCTable *previous = &table2;

	int factor = 1;
	int count = 0;

	while (fileNumber < 14) 
	{		
		downsample(&frame1);
		downsample(&frame2);	

		cout << endl << endl << endl << "Using " << imageName1 << " and " << imageName2 << endl;
		namedWindow("Frame 1", WINDOW_NORMAL);
		moveWindow("Frame 1", 50, 80);
		imshow("Frame 1", frame1);

		namedWindow("Frame 2", WINDOW_NORMAL);
		moveWindow("Frame 2", 400, 80);
		imshow("Frame 2", frame2);
		
		cout << "Matching..." << endl;
		MatchingPoints robustMatcher(frame1, frame2);
		vector<KeyPoint> keypoints1 = robustMatcher.getKeyPoints1();
		vector<KeyPoint> keypoints2 = robustMatcher.getKeyPoints2();
		vector<Colours> keyPointColours = robustMatcher.getColours(frame1);


		if (robustMatcher.hasEnoughMatches()) 
		{
			robustMatcher.displayFull(frame1, frame2);
			cout << "Enough Matches! " << endl;
			if (DEBUG == 0) 
			{
				cout << endl << keypoints1.size() << " matches!" << endl;				
			}
			//add into point cloud

			Mat K = matrixCalculator.findMatrixK(frame1);
			if (initial3dModel == true) 
			{
				cout << "Calculating initial camera matricies..." << endl;
				matrixCalculator.FindCameraMatrices(keypoints1, keypoints2, robustMatcher.getFundamentalMatrix(), P, P1, pointCloud);
				
				cout << "Creating initial 3D model..." << endl;
				pointCloud = matrixCalculator.triangulation(keypoints1, keypoints2, K, P, P1, pointCloud);
				(*current).addAllEntries(keypoints2, pointCloud);

				cout << "Initial Lookup table size is: " << current->tableSize() << endl;
				initial3dModel = false;
			} else 
			{
				cout << "Previous (current)  Table Size is " << current->tableSize() << endl;

				cout << "Previous (previous)  Table Size is " << previous->tableSize() << endl;
				
				//LUTable *temp;

				//previous->cleanup();
				//temp = previous;
				//previous = current;
				//current = temp;
				//current->cleanup();
				//current->init();

				
				//previous->cleanup();

				previous->init();
				previous = current;

				if(current == & table2)
				{
					current = &table1;

				} else 
				{
					current = &table2;
				}

				cout << "LookupTable Size is: " << previous->tableSize() << endl;
				cout << "New Table Size is: " << current->tableSize() << endl;

				P = P1; //images get shuffled along
				P1 = tableProcess(P1, keypoints2, keypoints1, current, previous, K);

				cout << "New Table Size after adding known 3d Points: " << current->tableSize() << endl;

				cout << "Triangulating..." << endl;
				pointCloud = matrixCalculator.triangulation(keypoints1, keypoints2, K, P, P1, pointCloud);
				current->addAllEntries(keypoints2, pointCloud);

				cout << "Table Size after adding Triangulated points: " << current->tableSize() << endl;
				
			}

			int numberOfPointsAdded = keypoints1.size();
			cout << "Start writing points to file..." << endl;
			//create new file each time we process features from a new image
			buildModel.insert_header(pointCloud.size(), fileNumber);

			//write previous points;
			for (int i = 0; i < previousNumberOfPointsAdded; i++) 
			{
				Point3d point = pointCloud.at(i).point;
				int blue = pointCloud.at(i).blue;
				int green = pointCloud.at(i).green;
				int red = pointCloud.at(i).red;
				buildModel.insert_point(point.x, point.y, point.z, red, green, blue, fileNumber);
			}

			//write current points			
			for (int i = 0; i < numberOfPointsAdded; i++) 
			{
				Point3d point = pointCloud.at(i + previousNumberOfPointsAdded).point;
				Colours pointColour = keyPointColours.at(i);
				pointCloud.at(i + previousNumberOfPointsAdded).blue = pointColour.blue;
				pointCloud.at(i + previousNumberOfPointsAdded).red = pointColour.red;
				pointCloud.at(i + previousNumberOfPointsAdded).green = pointColour.green;
				buildModel.insert_point(point.x, point.y, point.z, pointColour.red, pointColour.green, pointColour.blue, fileNumber);
			}

			fileNumber++;

			previousNumberOfPointsAdded = numberOfPointsAdded + previousNumberOfPointsAdded;

			cout << "End adding points" << endl;

		} else 
		{
			cout << "Not enough matches!" << endl;
		}

		//KeyPoint
		//for (int i = 0; i < keypoints1.size(); i++) {
		//	KeyPoint point1 = keypoints1.at(i);
		//	KeyPoint point2 = keypoints2.at(i);
		//	cout << i << ") (" << point1.pt.x << ", " << point1.pt.y << ") matches with (";
		//	cout << point2.pt.x << ", " << point2.pt.y << ")" << endl;
		//}

		cout << imageName1 << " " << imageName2 << " done,  Image is " << frame1.cols << "x" << frame1.rows << " " <<endl;

			pictureNumber1 = (pictureNumber2)%numberPictures;
			pictureNumber2 = (pictureNumber2 + factor)%numberPictures;
			char key = (char)waitKey(20);

			count++;
			cout << "Count is " << count << " Factor is " << factor << endl;
			if (count % (numberPictures) == numberPictures - 1) 
			{
				pictureNumber2++;
				factor++;
			}
		string stringpicturenumber1 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber1)) ->str();
		string stringpicturenumber2 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber2)) ->str();
		imageName1 = image + stringpicturenumber1 + extension;
		imageName2 = image + stringpicturenumber2 + extension;
		frame1 = imread(imageName1, -1);
		frame2 = imread(imageName2, -1);
	}

	cout << endl << "Done" << endl;
			char key = (char)waitKey(0);
		switch (key) {
			//case 27: //what is this for lol
		case 's':			
			break;
		case 'd':
			pictureNumber1 = (pictureNumber1 + 1)%numberPictures;
			pictureNumber2 = (pictureNumber2 + 1)%numberPictures;
			break;
		case 'r':
			//do it again
			break;
		case 'q':
			return 0;
			break;
		}
	return 0;
}

Matx34d tableProcess(Matx34d P1, 
	vector<KeyPoint> newKeyPoints, 
	vector<KeyPoint> oldKeyPoints, 
	PCTable *current, 
	PCTable *previous, 
	Mat K) 
{
	Point3d *found;
	vector<Point2d> foundPoints2d;
	vector<Point3d> foundPoints3d;
	vector<KeyPoint> newKeyPoints_notIn;
	vector<KeyPoint> oldKeyPoints_notIn;

	for (int i = 0; i < oldKeyPoints.size(); i++) 
	{
		found = previous->find_3d(oldKeyPoints.at(i).pt);

		if (found != NULL) 
		{
			Point3d newPoint;
			newPoint.x = found->x;
			newPoint.y = found->y;
			newPoint.z = found->z;
			Point2d newPoint2;
			newPoint2.x = newKeyPoints.at(i).pt.x;
			newPoint2.y = newKeyPoints.at(i).pt.y;
			foundPoints3d.push_back(newPoint);
			foundPoints2d.push_back(newPoint2);						
			current->add_entry(&newPoint, &newPoint2);
		}
	}

	//cout << foundPoints3d.size();
	cout << "Matches found in table: " << foundPoints2d.size() << endl;

	int size = foundPoints3d.size();

	//Mat found3dPoints = Mat::zeros(size,3, CV_64F);
	//Mat found2dPoints = Mat::zeros(size,2, CV_64F);

	Mat_<double> found3dPoints(size, 3);
	Mat_<double> found2dPoints(size, 2);

	for (int i = 0; i < size; i++)
	{

		found3dPoints(i, 0) = foundPoints3d.at(i).x;
		found3dPoints(i, 1) = foundPoints3d.at(i).y;
		found3dPoints(i, 2) = foundPoints3d.at(i).z;

		found2dPoints(i, 0) = foundPoints2d.at(i).x;
		found2dPoints(i, 1) = foundPoints2d.at(i).y;

	}

	Mat_<double> temp1(found3dPoints);
	Mat_<double> temp2(found2dPoints);

	Mat P(P1);

	Mat r(P, Rect(0, 0, 3, 3));
	Mat t(P, Rect(3, 0, 1, 3));

	Mat r_rog;
	cv::Rodrigues(r, r_rog);


	Mat dist = Mat::zeros(1, 4, CV_32F);
	double _dc[] = {0, 0, 0, 0};

	cv::solvePnP(found3dPoints, found2dPoints, K, Mat(1, 4, CV_64FC1, _dc), r_rog, t, false);

	cout << "Got new Camera matrix" << endl;
	
	Mat_<double> R1(3, 3);
	Mat_<double> t1(t);

	cv::Rodrigues(r_rog, R1);
	
	Mat camera = (Mat_<double> (3,4) << 	R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
		R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
		R1(2,0),	R1(2,1),	R1(2,2),	t1(2));

	return Matx34d(camera);
}


void nearest(Mat image1, Mat image2) 
{
	int minHessian = 1000;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints1, keypoints2;

	detector.detect(image1, keypoints1);
	detector.detect(image2, keypoints2);

	SurfDescriptorExtractor extractor;

	Mat descriptors1, descriptors2;

	extractor.compute(image1, keypoints1, descriptors1);
	extractor.compute(image2, keypoints2, descriptors2);

	BruteForceMatcher<L2<float>> matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	int matchAmount = 200;

	std::nth_element(matches.begin(), matches.begin() + matchAmount, matches.end());
	matches.erase(matches.begin() + matchAmount, matches.end());


	Mat imageMatches;
	drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches);
	namedWindow("Compare image", WINDOW_NORMAL);
	//imshow("Compare image", imageMatches);
}

void downsample(Mat *image) 
{
	int maxRows = 1800;
	int maxCols = 1600;
	Mat modifyImage = *image;
	int height = modifyImage.rows;
	int width = modifyImage.cols;

	//account for odds
	if (height%2 != 0) 
	{
		height--;
	}
	if (width%2 != 0)
	{
		width--;
	}
	//form new images:
	Mat evenSize(modifyImage, Rect(0, 0, width - 1, height - 1));
	Mat downSize;
	while (height * width > maxRows * maxCols) 
	{ 
		pyrDown(evenSize, downSize, Size(width/2, height/2));
		//set new image to the downsized one
		*image = downSize;
		//do again and account for odds
		height = downSize.rows;
		width = downSize.cols;

		if (height%2 != 0) 
		{
			height--;
		}
		if (width%2 != 0) 
		{
			width--;
		}
		Mat next(downSize, Rect(0, 0, width - 1, height - 1));
		evenSize = next;		
	}	
}
