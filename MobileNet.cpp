#include "MobileNet.h"

void ltrim(std::string& s) {
	s.erase(s.begin(), find_if(s.begin(), s.end(), [](char ch) {
		return !isspace(ch);
		}));
}

// trim from end (in place)
void rtrim(std::string& s) {
	s.erase(find_if(s.rbegin(), s.rend(), [](char ch) {
		return !isspace(ch);
		}).base(), s.end());
}

// trim from both ends (in place)
void trim(std::string& s) {
	ltrim(s);
	rtrim(s);
}

int split(const std::string& str, std::vector<std::string>& ret_, std::string sep) {
	if (str.empty()) {
		return 0;
	}

	std::string tmp;
	std::string::size_type pos_begin = str.find_first_not_of(sep);
	std::string::size_type comma_pos = 0;

	while (pos_begin != std::string::npos) {
		comma_pos = str.find(sep, pos_begin);
		if (comma_pos != std::string::npos) {
			tmp = str.substr(pos_begin, comma_pos - pos_begin);
			pos_begin = comma_pos + sep.length();
		}
		else {
			tmp = str.substr(pos_begin);
			pos_begin = comma_pos;
		}

		if (!tmp.empty()) {
			trim(tmp);
			ret_.push_back(tmp);
			tmp.clear();
		}
	}
	return 0;
}

int split(const std::string& str, std::vector<int>& ret_, std::string sep) {
	std::vector<std::string> tmp;
	split(str, tmp, sep);

	for (int i = 0; i < tmp.size(); i++) {
		ret_.push_back(std::stoi(tmp[i]));
	}
	return ret_.size();
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> ReadLabels(const std::string path,int size) {
	std::ifstream fs;
	fs.open(path.c_str(), std::ios::in);
	std::string linestring;
	
	torch::Tensor line;
	torch::Tensor label;
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> returnvalue;
	while (getline(fs, linestring)) {
		trim(linestring);
		if (linestring.empty()) {
			continue;
		}
		std::vector<std::string> ret;
		split(linestring, ret, " ");
		
		if (ret.size() == 2) 
		{
			std::string p_key = ret[0];
			std::string p_value = ret[1];
			auto image=cv::imread(p_key);
			cv::resize(image,image,cv::Size(size, size));
			
			line = torch::from_blob(image.data, { image.rows, image.cols ,3 }, torch::kByte).toType(torch::kFloat);
			line = line.permute({ 2,0,1 });
			
			label = torch::tensor({ std::stoi(p_value) }).toType(torch::kLong);
			returnvalue.push_back(std::make_tuple(line, label));
		}
	}
	fs.close();
	return returnvalue;
}

void train_MobileNetv3() {
	DWORD time;
	torch::manual_seed(1);
	size_t epochs = 50;
	size_t batch_size = 20;//批大小
	size_t test_batch_size = 20;
	float lr = 0.001;
	float momentum = 0.9;

	std::string trainlabels = "models/simpleconv3/datas/train/train_labels.txt";
	std::string testlabels = "models/simpleconv3/datas/test/test_labels.txt";
	auto device_type = torch::kCUDA;
	auto device = torch::Device(device_type);
	std::cout << "开始创建模型" << std::endl;

	auto net = std::make_shared<MobileNetV3>(2);
	
	net->to(device);
	std::cout << "模型创建成功" << std::endl;
	auto train_dataset = SimpleDataset(trainlabels,224).map(torch::data::transforms::Stack<>());

	const auto dataset_size = train_dataset.size().value();
	auto train_loader = torch::data::make_data_loader(std::move(train_dataset), batch_size);
	auto test_dataset = SimpleDataset(testlabels,224).map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader = torch::data::make_data_loader(std::move(test_dataset), test_batch_size);
	//torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(lr).momentum(momentum));
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr).betas({ 0.9,0.999 }));
	std::cout << "开始训练" << std::endl;
	auto t1 = std::chrono::steady_clock::now();
	for (size_t epoch = 1; epoch <= epochs; ++epoch) {
		trainMobileNetv3(epoch, batch_size, net, device, *train_loader, optimizer, dataset_size);
		testMobileNetv3(test_batch_size, net, device, *test_loader, test_dataset_size);
	}
	std::string savepath = "models/MobileNetv3.pt";
	torch::save(net, savepath);
	auto t2 = std::chrono::steady_clock::now();
	std::cout << "耗时(秒):" << std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000 << std::endl;
}

void test_MobileNetv3() {
	auto t1 = std::chrono::steady_clock::now();
	auto device_type = torch::kCPU;
	torch::Device auto_device(device_type);
	std::string savepath = "models/MobileNetv3.pt";
	auto net2 = std::make_shared<MobileNetV3>();
	torch::load(net2, savepath);
	net2->to(auto_device);
	net2->eval();

	torch::jit::Module module2 = torch::jit::load("models/dbfacelibtorch.pt");
	auto net = std::make_shared<torch::jit::Module>(module2);
	net->to(auto_device);
	net->eval();

	std::string imgpath = "models/12_Group_Group_12_Group_Group_12_728.jpg";
	auto im = cv::imread(imgpath);
	auto im_show = cv::imread(imgpath);
	auto objs = JitDBFaceDetect(net, im, 0.2);

	cv::Mat roi;
	cv::Mat roiresized;
	int testsize = 224;
	float mean[3] = { 0.5, 0.5, 0.5 };
	float std[3] = { 0.5, 0.5, 0.5 };
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	int i;
	for (int K = 0; K < objs.size(); ++K) {
		auto& obj = objs[K];

		ymin = obj.landmark[2].y;
		ymax = obj.landmark[3].y + obj.landmark[3].y - obj.landmark[2].y;
		xmin = obj.landmark[3].x;
		xmax = obj.landmark[4].x;

		if (ymax > im.rows) {
			ymax = im.rows - 1;
		}
		if (ymin > im.rows) {
			ymin = im.rows - 1;
		}
		if (xmax > im.cols) {
			xmax = im.cols - 1;
		}
		if (xmin > im.cols) {
			xmin = im.cols - 1;
		}
		if (ymin < 0) {
			ymin = 0;
		}
		if (ymax < 0) {
			ymax = 0;
		}
		if (xmin < 0) {
			xmin = 0;
		}
		if (xmax < 0) {
			xmax = 0;
		}
		if (ymin == ymax) {
			ymax = ymin + 1;
		}

		if (xmin == xmax) {
			xmax = xmin + 1;
		}

		roi = im(cv::Range(ymin, ymax), cv::Range(xmin, xmax));

		cv::resize(roi, roiresized, cv::Size(testsize, testsize));
		auto img_tensor = torch::from_blob(roiresized.data, { 1, roiresized.rows, roiresized.cols, 3 }, torch::kByte).toType(torch::kFloat).permute({ 0,3 , 1, 2 }).to(auto_device);

		torch::Tensor out_tensor = net2->forward(img_tensor);

		
		auto index = out_tensor.argmax(1);
		cv::rectangle(im_show, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 255), 2);
		if (index.item<float>() == 0) {
			cv::putText(im_show, "none", cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}
		else {
			cv::putText(im_show, "smile", cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}
	}
	auto t2 = std::chrono::steady_clock::now();
	std::cout << "耗时(毫秒):" << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;
	cv::namedWindow("result", 0);
	cv::imshow("result", im_show);
	cv::waitKey(0);

}