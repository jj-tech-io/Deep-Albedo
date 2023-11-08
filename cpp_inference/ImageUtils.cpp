#include "ImageUtils.h"
#

cv::Mat ImageUtils::load_image_as_mat(string image_path, int width, int height)
{
	//load image
	cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
	//convert to float
	image.convertTo(image, CV_32FC3);
	//resize
	resize(image, image, cv::Size(width, height));
	//normalize
	image /= 255.0f;
	return image;
}

void ImageUtils::display_mat_image(cv::Mat image, std::string window_label, bool gamma = true)
{
	// Normalize image to range [0,1] for display
	cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_32FC3);
	// If you want to print out max and min values:
	double min, max;
	cv::minMaxLoc(image, &min, &max);
	cout << "min: " << min << " max: " << max << endl;
	// Convert to 8-bit unsigned integer for display
	image.convertTo(image, CV_8UC3);
	int original_width = image.cols;
	int original_height = image.rows;
	window_label = window_label;
	// Display the image
	cv::resize(image, image, cv::Size(256,256));
	//gamma correction
	if (gamma) {
		cv::Mat gamma_corected;
		double av = (1.1 + 2.2) / 2;
		image.convertTo(gamma_corected, CV_8UC3, 1.0 / av);
		cv::imshow(window_label, gamma_corected);
	}
	else {
		cv::imshow(window_label, image);
	}
}

vector<vector<vector<float>>> ImageUtils::image_mat_to_vector(const Mat& image)
{
    // Check if the image type is CV_32F with 3 channels
    if (image.type() != CV_32FC3) {
        throw std::runtime_error("Unsupported image format: Expected CV_32FC3.");
    }

    // Convert to vector
    vector<vector<vector<float>>> image_vector(image.rows, vector<vector<float>>(image.cols, vector<float>(image.channels())));
    for (int i = 0; i < image.rows; i++) {
        // Get pointer to the start of row i
        const Vec3f* row_ptr = image.ptr<Vec3f>(i);
        for (int j = 0; j < image.cols; j++) {
            // Here we assume the channels are in BGR order and we need to swap them to RGB
            Vec3f bgr = row_ptr[j];
            image_vector[i][j][0] = bgr[2]; // Blue channel
            image_vector[i][j][1] = bgr[1]; // Green channel
            image_vector[i][j][2] = bgr[0]; // Red channel
        }
    }
    return image_vector;
}


//3d to 2d
vector<vector<float>> ImageUtils::im_vector_3d_to_2d(vector<vector<vector<float>>> image, int channels = 3)
{
	//convert to vector
	vector<vector<float>> image_matrix(image.size() * image[0].size(),vector<float>(image[0][0].size()));
	for (int i = 0; i < image.size(); i++) {
		for (int j = 0; j < image[0].size(); j++)
		{
			for (int k = 0; k < image[0][0].size(); k++)
			{
				//this switches the order of the channels from BGR to RGB
				image_matrix[i * image[0].size() + j][k] = image[i][j][k];
			}
		}
	}
	return image_matrix;
}
//2d to 3d
vector<vector<vector<float>>> ImageUtils::im_vector_2d_to_3d(vector<vector<float>> image_2d, int channels = 3)
{
	// Safety checks
	if (image_2d.empty()) {
		cerr << "Error: image_2d is empty!" << endl;
		return {};  // Return an empty vector
	}
	// Assuming the width of the image is the same as the number of channels in the 2D representation.
	int width = sqrt(image_2d.size());
	int height = width; // Assuming a square image, you may need to adjust if it's a different shape.
	cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << endl;  // Debug output
	vector<vector<vector<float>>> image_3d(height, vector<vector<float>>(width, vector<float>(channels)));
	for (int i = 0; i < image_2d.size(); i++)
	{
		int row = i / width;
		int col = i % width;
		if (row >= height || col >= width) {
			cerr << "Error: Invalid indices. Row: " << row << ", Col: " << col << endl;
			return {};
		}
		for (int k = 0; k < channels; k++)
		{
			image_3d[row][col][k] = image_2d[i][k];
		}
	}
	return image_3d;
}





void ImageUtils::display_vector_image(vector<vector<vector<float>>> image, int channels = 3, std::string window_label="")
{
	cv::Mat image_mat(image.size(), image[0].size(), CV_32FC3);
	for (int i = 0; i < image_mat.rows; i++) {
		for (int j = 0; j < image_mat.cols; j++)
		{
			for (int k = 0; k < image_mat.channels(); k++)
			{
					//this switches the order of the channels from RGB to BGR
					image_mat.at<cv::Vec3f>(i, j)[k] = image[i][j][k];
			}
		}
	}
	//display
	display_mat_image(image_mat, window_label);
}

vector<vector<vector<float>>> ImageUtils::un_flatten_image(vector<float> flattened_image, int height, int width, int channels)
{
	vector<vector<vector<float>>> image_3d(height, vector<vector<float>>(width, vector<float>(channels)));

	int index = 0;  // Pointer for the flattened_image

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				if (index >= flattened_image.size()) {
					cerr << "Error: Exceeded flattened image size at index: " << index << endl;
					return {};
				}
				image_3d[i][j][k] = flattened_image[index++];
			}
		}
	}
	return image_3d;
}

vector<float> ImageUtils::flatten_image(vector<vector<vector<float>>> image_3d)
{
	vector<float> flattened_image;
	//check if 0 to 1 or 0 to 255

	for (int i = 0; i < image_3d.size(); i++) {
		for (int j = 0; j < image_3d[i].size(); j++) {
			for (int k = 0; k < image_3d[i][j].size(); k++) {
				
				flattened_image.push_back(image_3d[i][j][k]);
			}
		}
	}
	return flattened_image;
}

std::vector<cv::Mat> ImageUtils::vector_to_mats(const std::vector<std::vector<std::vector<float>>>& vec) {
	if (vec.empty() || vec[0].empty() || vec[0][0].empty()) {
		// Handle the error appropriately.
		// For now, just return an empty vector of Mats.
		return std::vector<cv::Mat>();
	}

	// Assuming the vector has the format [height][width][channels].
	int channels = vec.size();
	int height = vec[0].size();
	int width = vec[0][0].size();
	std::cout << "height: " << height << " width: " << width << " channels: " << channels << endl;
	// Create a vector of Mats to hold each channel.
	std::vector<cv::Mat> channel_mats(channels, cv::Mat(height, width, CV_32FC1));
	float max = 0.0f;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			for (int k = 0; k < channels; ++k) {
				// The vector already contains values in the range [0,1], no need to normalize.
				double value = vec[k][i][j];
				if (value > max) {
					max = value;
				}

				// Assign to the corresponding Mat.
				channel_mats[k].at<float>(i, j) = value;
			}
		}
	}
	return channel_mats;
}




vector<vector<vector<float>>> ImageUtils::swap_dim_0_and_2(vector<vector<vector<float>>> image)
{
	int height = image.size();
	int width = image[0].size();
	int channels = image[0][0].size();
	//convert to vector
	vector<vector<vector<float>>> swapped_image(image[0].size(), vector<vector<float>>(image.size(), vector<float>(image[0][0].size())));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				swapped_image[k][i][j] = image[i][j][k];
			}
		}
	}
	return swapped_image;
}

vector<float> ImageUtils::normalize_vector(vector<float> vec)
{
	vector<float> normalized_image(vec.size());
	//get max and min  of image
	float max = *max_element(vec.begin(), vec.end());	
	float min = *min_element(vec.begin(), vec.end());
	for (int i = 0; i < vec.size(); i++) {
		normalized_image[i] = (vec[i] - min) / (max - min);
	}
	return normalized_image;
}

vector<float> ImageUtils::normalize_vec_255(vector<float> vec)
{
	vector<float> unnormalized_image(vec.size());

	float max = *max_element(vec.begin(), vec.end());
	float min =  *min_element(vec.begin(), vec.end());
	std::cout << "max: " << max << " min: " << min << endl;
	for (int i = 0; i < vec.size(); i++) {
		vec[i] =  vec[i] * (max - min) + min;
		vec[i] *= 255;
		if (vec[i] > max) {
			vec[i] = max;
		}
		if (vec[i] < min) {
			vec[i] = min;
		}
	}
	max = *max_element(vec.begin(), vec.end());
	min = *min_element(vec.begin(), vec.end());
	std::cout << "max: " << max << " min: " << min << endl;
	return vec;
}
