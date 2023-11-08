#include "ImageUtils.h"
#include <opencv2/core/matx.hpp>
#include <onnxruntime_cxx_api.h>
constexpr int64_t width = 512;
constexpr int64_t height = 512;
constexpr int64_t batch_size = width * height;

std::vector<std::vector<std::vector<float>>> encode(cv::Mat image, bool info_print = false) {
	ImageUtils image_utils;
	constexpr int64_t inChannels = 3;
	constexpr int64_t outChannels = 5;
	Ort::Env encoder_env;
	Ort::RunOptions runOptions;
	Ort::Session encoder_session(nullptr);
	Ort::SessionOptions ort_session_options;
	OrtCUDAProviderOptions options;
	options.device_id = 0;
	auto r = OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);
	auto encoder_path = L"C:\\Users\\joeli\\Dropbox\\Code\\AE_2023_06\\TrainedModels\\encoder_sep_7.onnx";
	encoder_env = Ort::Env();
	encoder_session = Ort::Session(encoder_env, encoder_path, ort_session_options);
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr encoder_input_name = encoder_session.GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr encoder_output_name = encoder_session.GetOutputNameAllocated(0, ort_alloc);
	const std::array<const char*, 1> encoder_input_names = { encoder_input_name.get() };
	const std::array<const char*, 1> encoder_output_names = { encoder_output_name.get() };
	encoder_input_name.release();
	encoder_output_name.release();
	const std::array<int64_t, 2> encoder_input_shape = { 1,inChannels };
	const std::array<int64_t, 2> encoder_output_shape = { 1, outChannels };
	std::vector<std::vector<std::vector<float>>> encoder_output_data(outChannels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
	// Prepare all pixels' RGB data at once for batching
	std::vector<float> flattenedEncoderInput(width * height * inChannels);
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			int idx = (h * width + w) * inChannels;
			for (int i = 0; i < inChannels; ++i) {
				flattenedEncoderInput[idx + i] = image.at<cv::Vec3f>(h, w)[i];
			}
		}
	}
	std::vector<int64_t> batched_input_shape = { height * width, inChannels };  // Adjust shape for batch processing
	std::vector<int64_t> batched_output_shape = { height * width, outChannels }; // Adjust shape for batch processing
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, flattenedEncoderInput.data(), flattenedEncoderInput.size(), batched_input_shape.data(), batched_input_shape.size());
	std::vector<float> batchedOutput(width * height * outChannels); // Container for batched output
	auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, batchedOutput.data(), batchedOutput.size(), batched_output_shape.data(), batched_output_shape.size());
	try {
		encoder_session.Run(runOptions, encoder_input_names.data(), &inputTensor, 1, encoder_output_names.data(), &outputTensor, 1);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
	}
	// Process the batched output to the desired data structure
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			for (int i = 0; i < outChannels; ++i) {
				encoder_output_data[i][h][w] = batchedOutput[(h * width + w) * outChannels + i];
			}
		}
	}
	if (info_print) {
		std::cout << "Input name: " << encoder_input_name.get() << std::endl;
		std::cout << "Output name: " << encoder_output_name.get() << std::endl;
		std::cout << "encoder_output_data.size: " << encoder_output_data.size() << std::endl;
		std::cout << "encoder_output_data[0].size: " << encoder_output_data[0].size() << std::endl;
		std::cout << "encoder_output_data[0][0].size: " << encoder_output_data[0][0].size() << std::endl;
	}
	return encoder_output_data;
}
cv::Mat decoder(std::vector<std::vector<std::vector<float>>> parameter_maps, bool info_print = false) {
	ImageUtils image_utils;
	constexpr int64_t inChannels = 5;
	constexpr int64_t outChannels = 3;
	constexpr int64_t decoder_input_elements = inChannels * height * width;
	constexpr int64_t decoder_output_elements = outChannels * height * width;
	Ort::Env decoder_env;
	Ort::RunOptions runOptions;
	Ort::Session decoder_session(nullptr);
	//auto decoder_path = L"C:\\Users\\joeli\\OneDrive\\Documents\\GitHub\\python_projects\\AE_Log\\onnx_decoder.onnx";
	//C:\Users\joeli\Dropbox\Code\AE_2023_06\decoder_sep_7.onnx
	auto decoder_path = L"C:\\Users\\joeli\\Dropbox\\Code\\AE_2023_06\\TrainedModels\\decoder_sep_7.onnx";
	// Use CUDA GPU
	Ort::SessionOptions decoder_ort_session_options;
	OrtCUDAProviderOptions decoder_options;
	decoder_options.device_id = 0;
	auto r = OrtSessionOptionsAppendExecutionProvider_CUDA(decoder_ort_session_options, decoder_options.device_id);
	decoder_env = Ort::Env();
	decoder_session = Ort::Session(decoder_env, decoder_path, decoder_ort_session_options);
	Ort::AllocatorWithDefaultOptions decoder_ort_alloc;
	Ort::AllocatedStringPtr decoder_input_name = decoder_session.GetInputNameAllocated(0, decoder_ort_alloc);
	Ort::AllocatedStringPtr decoder_output_name = decoder_session.GetOutputNameAllocated(0, decoder_ort_alloc);
	const std::array<const char*, 1> decoder_input_names = { decoder_input_name.get() };
	const std::array<const char*, 1> decoder_output_names = { decoder_output_name.get() };
	decoder_input_name.release();
	decoder_output_name.release();
	const std::array<int64_t, 2> decoder_input_shape = { 1,inChannels };
	const std::array<int64_t, 2> decoder_output_shape = { 1, outChannels };
	std::vector<float> decoder_input_tensor_vals(decoder_input_elements);
	std::vector<float> decoder_output_tensor_vals(decoder_output_elements);
	//std::copy_n(decoder_input_vector.begin(), decoder_input_elements, decoder_input_tensor_vals.begin());
	std::vector<std::vector<std::vector<float>>> decoder_output_data(outChannels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
	std::array<float, outChannels> decoderOutPixel{};
	cv::Mat decoderOutMat = cv::Mat(decoder_output_data[0].size(), decoder_output_data[0][0].size(), CV_32FC3);
	// Prepare all pixels' data at once for batching
	std::vector<float> flattenedDecoderOutput(width * height * inChannels); // Note that 'inChannels' is now 5 based on your provided code
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			int idx = (h * width + w) * inChannels;
			for (int i = 0; i < inChannels; ++i) {
				flattenedDecoderOutput[idx + i] = parameter_maps[i][h][w];
			}
		}
	}
	std::vector<int64_t> batched_decoder_input_shape = { height * width, inChannels };  // Adjust shape for batch processing
	std::vector<int64_t> batched_decoder_output_shape = { height * width, outChannels }; // Adjust shape for batch processing
	auto decoder_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto decoder_input_tensor = Ort::Value::CreateTensor<float>(decoder_memory_info, flattenedDecoderOutput.data(), flattenedDecoderOutput.size(), batched_decoder_input_shape.data(), batched_decoder_input_shape.size());
	std::vector<float> batchedDecoderOutput(width * height * outChannels); // Container for batched output
	auto decoder_output_tensor = Ort::Value::CreateTensor<float>(decoder_memory_info, batchedDecoderOutput.data(), batchedDecoderOutput.size(), batched_decoder_output_shape.data(), batched_decoder_output_shape.size());
	try {
		decoder_session.Run(runOptions, decoder_input_names.data(), &decoder_input_tensor, 1, decoder_output_names.data(), &decoder_output_tensor, 1);
	}
	catch (Ort::Exception& e) {
		std::cout << "Decoder exception:";
		std::cout << e.what() << std::endl;
	}
	// Process the batched output to the desired data structures
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			for (int i = 0; i < outChannels; ++i) {
				float value = batchedDecoderOutput[(h * width + w) * outChannels + i];
				//decoder_output_data[i][h][w] = value;
				decoderOutMat.at<cv::Vec3f>(h, w)[i] = value * 255;
			}
		}
	}
	//print out channel dimensions
	if (info_print) {
		std::cout << "Input name: " << decoder_input_name.get() << std::endl;
		std::cout << "Output name: " << decoder_output_name.get() << std::endl;
		std::cout << "decoder_output_data size: " << decoder_output_data.size() << std::endl;
		std::cout << "decoder_output_data[0] size: " << decoder_output_data[0].size() << std::endl;
		std::cout << "decoder_output_data[0][0] size: " << decoder_output_data[0][0].size() << std::endl;
		std::cout << "decoder_input_shape: " << decoder_input_shape[0] << " " << decoder_input_shape[1] << std::endl;
	}
	cv::normalize(decoderOutMat, decoderOutMat, 0, 255, cv::NORM_MINMAX, CV_8U);
	return decoderOutMat;
}

int main()
{
	bool info_print = false;
	//std::string image_path = "C:\\Users\\joeli\\Dropbox\\AE_MC\\AE_InputModels\\test_im.png";
	std::string image_path = "C:\\Users\\joeli\\Dropbox\\AE_MC\\AE_InputModels\\m46.png";
	ImageUtils image_utils;
	cv::Mat image = image_utils.load_image_as_mat(image_path, width, height);
	image_utils.display_mat_image(image, "Original Image", true);
	cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

	/*---------------Encode----------------------------------------------*/
	//start timer for encode
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<std::vector<float>>> parameter_maps = encode(image, false);
	//end timer for encode
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "encode time: " << elapsed.count() << std::endl;

	std::vector<cv::Mat> parameter_maps_mats = image_utils.vector_to_mats(parameter_maps);
	// Display the parameter maps
	image_utils.display_mat_image(parameter_maps_mats[0], "Cm", false);
	image_utils.display_mat_image(parameter_maps_mats[1], "Ch", false);
	image_utils.display_mat_image(parameter_maps_mats[2], "Bm", false);
	image_utils.display_mat_image(parameter_maps_mats[3], "Bh", false);
	image_utils.display_mat_image(parameter_maps_mats[4], "T", false);


	/*---------------Decoder----------------------------------------------*/
	//start timer for decoder
	start = std::chrono::high_resolution_clock::now();
	cv::Mat decoder_out = decoder(parameter_maps, false);

	// end timer for decoder
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "decode time: " << elapsed.count() << std::endl;

	// Display the decoded image
	image_utils.display_mat_image(decoder_out, "Decoded Image",true);

	//destroy windows
	cv::waitKey(0);
	return 0;
}
		