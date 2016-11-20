#include <iostream>
#include <fstream>
using namespace std;

class buildModel
{
public:
	void insert_point(double x, double y, double z, int b, int g, int r, int index);
	void insert_header(int pointCloudSize, int index);

};

