#include <iostream> 
#include <string>  
#include <iomanip>  
#include <sstream>  
#include <vector>


#include "opencv/cv.h"
#include "dataTypeStructs.h"

using namespace std;
using namespace cv;


typedef struct PCT_STRUCT{
	Point2d *fp;
	Point3d *cp;
}Entry;


class PCTable {
  public:
	PCTable();
	void init();
    void add_entry(Point3d *cp, Point2d *fp);  // adding to the end of the list
	Point3d* find_3d(Point2d fp);				//GET 3D coord given 2D
	void cleanup();
	bool test();
	void addAllEntries(vector<KeyPoint> twoDee, vector<SpacePoint> threeDee);
	int tableSize();

  private:
	vector <Entry> table;				//storage of entries
	int entry_num;
	
};