#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <getopt.h>
#include <boost/program_options.hpp>
#include <bitset>
#include <string>
#include <array>

class Cost
{
public:
    Cost() {}                                                //empty constructor
    virtual int apply(const int &x, const int &y) const = 0; //pure virtual function, derived class will implement it
};

class AbsoluteDifferenceCost : public Cost
{

public:
    int ADLookUpTable[256][256];
    int process(const int &x, const int &y) const;
// public:
    AbsoluteDifferenceCost();
    int apply(const int &x, const int &y) const override;
};

class CensusCost : public Cost
{
public:
    int censusLookUpTable[256][256];
    int process(const int &x, const int &y) const;
// public:
    CensusCost();
    int apply(const int &x, const int &y) const override;
};
class StereoMatching
{
public:
    StereoMatching(std::string &left, std::string &right, int &d_min, int &d_max,
                   int &radius, Cost &cost);
    cv::Mat getDisparityMap();
    void setNeedCensusTransform(bool flag) {needCensusTransform = flag;}
private:
    cv::Mat gray1, gray2, outputMap;
    int d_min, d_max, radius, window, disparityRange, rows, columns;
    Cost &cost;
    std::vector<std::unique_ptr<cv::Mat>> results;
    bool needCensusTransform = 0;
    void prepareCostPerDisparity();
    void rollingWindowCost();
    void preprocessImages();
    cv::Mat censusTransform(const cv::Mat &matrix, int window);
};

//helper methods
void showImage(const cv::Mat &image, std::string windowName);
void destroyWindows();
// cv::Mat censusTransform(const cv::Mat &matrix, int window);