#pragma once
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <deque>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
class ImageUtils
{
public: 
	std::vector<cv::Mat> images;
	std::vector<std::string> image_names;
	std::vector<std::string> image_paths;
	map<string, int> image_name_to_index;
	cv::Mat load_image_as_mat(string image_path, int width, int height);
	void display_mat_image(cv::Mat image, std::string window_label, bool gamma);
	vector<vector<vector<float>>> image_mat_to_vector(const Mat& image);
	vector<vector<float>> im_vector_3d_to_2d(vector<vector<vector<float>>> image, int channels);
	vector<vector<vector<float>>> im_vector_2d_to_3d(vector<vector<float>> image, int channels);
	void display_vector_image(vector<vector<vector<float>>> image, int channels, std::string window_label);
	vector<vector<vector<float>>> un_flatten_image(vector<float> flattened_image, int height, int width, int channels);
	vector<float> flatten_image(vector<vector<vector<float>>> image_3d);
	std::vector<cv::Mat> vector_to_mats(const std::vector<std::vector<std::vector<float>>>& vec);
	vector < vector<vector<float>>> swap_dim_0_and_2(vector < vector<vector<float>>> image);
	//normalize
	vector<float> normalize_vector(vector<float> image);
	//unnormalize
	vector<float> normalize_vec_255(vector<float> image);
};

