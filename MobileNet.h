#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

void ltrim(std::string& s);
void rtrim(std::string& s);
void trim(std::string& s);
int split(const std::string& str, std::vector<std::string>& ret_, std::string sep);
int split(const std::string& str, std::vector<int>& ret_, std::string sep);
std::vector<std::tuple<torch::Tensor, torch::Tensor>> ReadLabels(const std::string path,int size);
void train_MobileNetv3();
void test_MobileNetv3();

struct HSwishImpl : torch::nn::Module {

	HSwishImpl() = default;

	torch::Tensor forward(torch::Tensor x) {
		auto y = x;
		y = y + 3;
		y = torch::nn::functional::relu6(y, torch::nn::functional::ReLU6FuncOptions(true));
		y = x * y;
		y = y / 6;
		return y;
	}
};

TORCH_MODULE(HSwish);

struct HSigmoidImpl : torch::nn::Module {

	HSigmoidImpl() = default;

	torch::Tensor forward(torch::Tensor x) {
		auto y = x;
		y = y + 3;
		y = torch::nn::functional::relu6(y, torch::nn::functional::ReLU6FuncOptions(true));
		y = y / 6;
		return y;
	}
};

TORCH_MODULE(HSigmoid);

struct SeModuleImpl : torch::nn::Module {
	SeModuleImpl(int in_size, int reduction = 4) :
		
		se(
			torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1,1 })),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size, in_size / reduction, { 1,1 }).bias(false)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_size / reduction)),
			torch::nn::ReLU(torch::nn::functional::ReLUFuncOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size / reduction, in_size, { 1,1 }).bias(false)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_size)),
			HSigmoid()
		)
	{
		
		register_module("se", se);
	};

	torch::Tensor forward(torch::Tensor x) {
		auto y = se->forward(x);
		return x * y;
	}
	
	torch::nn::Sequential se;
};

TORCH_MODULE(SeModule);


struct BlockHSwishModule : torch::nn::Module {
	BlockHSwishModule(int kernel_size, int in_size, int expand_size, int out_size, HSwish nolinear, SeModule semodule, int stride) :

		stride_(stride),
		in_size_(in_size),
		out_size_(out_size),
		se(semodule),
		conv1(torch::nn::Conv2dOptions(in_size, expand_size, { 1,1 }).bias(false)),
		bn1(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear1(nolinear),
		conv2(torch::nn::Conv2dOptions(expand_size, expand_size, { kernel_size,kernel_size}).stride({ stride,stride }).padding({ kernel_size / 2,kernel_size / 2 }).groups(expand_size).bias(false)),
		bn2(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear2(nolinear),
		conv3(torch::nn::Conv2dOptions(expand_size, out_size, { 1,1 }).bias(false)),
		bn3(torch::nn::BatchNorm2dOptions(out_size))
	{
		if (stride == 1 && in_size != out_size) {
			shortcut = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size, out_size, { 1,1 }).bias(false)), torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_size)));
		}
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("nolinear1", nolinear1);
		register_module("nolinear2", nolinear2);
		register_module("se", se);
		register_module("shortcut", shortcut);

		
	}
	torch::Tensor forward(torch::Tensor x) {
		auto y = nolinear1->forward(bn1->forward(conv1->forward(x)));
		
		y = nolinear2->forward(bn2->forward(conv2->forward(y)));
		
		y = bn3->forward(conv3->forward(y));
		
		y = se->forward(y);
		if (stride_ == 1 && in_size_ != out_size_) {
			
			y = y+ shortcut->forward(x);
		}
		else if (stride_ == 1 && in_size_ == out_size_) {
			y = y + x;
		}
		return y;
	}
	int stride_;
	int in_size_;
	int out_size_;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	HSwish nolinear1;
	HSwish nolinear2;
	torch::nn::BatchNorm2d bn1;
	torch::nn::BatchNorm2d bn2;
	torch::nn::BatchNorm2d bn3;
	SeModule se;
	torch::nn::Sequential shortcut;
};



struct BlockHSwishNullModule : torch::nn::Module {
	BlockHSwishNullModule(int kernel_size, int in_size, int expand_size, int out_size, HSwish nolinear,  int stride) :

		stride_(stride),
		in_size_(in_size),
		out_size_(out_size),
		conv1(torch::nn::Conv2dOptions(in_size, expand_size, { 1,1 }).bias(false)),
		bn1(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear1(nolinear),
		conv2(torch::nn::Conv2dOptions(expand_size, expand_size, { kernel_size,kernel_size }).stride({ stride,stride }).padding({ kernel_size / 2,kernel_size / 2 }).groups(expand_size).bias(false)),
		bn2(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear2(nolinear),
		conv3(torch::nn::Conv2dOptions(expand_size, out_size, { 1,1 }).bias(false)),
		bn3(torch::nn::BatchNorm2dOptions(out_size))
	{
		if (stride == 1 && in_size != out_size) {
			shortcut = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size, out_size, { 1,1 }).bias(false)), torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_size)));
		}
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("nolinear1", nolinear1);
		register_module("nolinear2", nolinear2);
		register_module("shortcut", shortcut);
	}
	torch::Tensor forward(torch::Tensor x) {
		auto y = nolinear1->forward(bn1->forward(conv1->forward(x)));
		y = nolinear2->forward(bn2->forward(conv2->forward(y)));
		y = bn3->forward(conv3->forward(y));

		if (stride_ == 1 && in_size_ != out_size_) {
			y = y + shortcut->forward(x);
		}
		else if (stride_ == 1 && in_size_ == out_size_) {
			y = y + x;
		}

		return y;
	}
	int stride_;
	int in_size_;
	int out_size_;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	HSwish nolinear1;
	HSwish nolinear2;
	torch::nn::BatchNorm2d bn1;
	torch::nn::BatchNorm2d bn2;
	torch::nn::BatchNorm2d bn3;

	torch::nn::Sequential shortcut;
};



struct BlockReLUModule : torch::nn::Module {
	BlockReLUModule(int kernel_size, int in_size, int expand_size, int out_size, torch::nn::ReLU nolinear, SeModule semodule, int stride) :

		stride_(stride),
		in_size_(in_size),
		out_size_(out_size),
		se(semodule),
		conv1(torch::nn::Conv2dOptions(in_size, expand_size, { 1,1 }).bias(false)),
		bn1(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear1(nolinear),
		conv2(torch::nn::Conv2dOptions(expand_size, expand_size, { kernel_size,kernel_size }).stride({ stride ,stride }).padding({ kernel_size / 2,kernel_size / 2 }).groups(expand_size).bias(false)),
		bn2(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear2(nolinear),
		conv3(torch::nn::Conv2dOptions(expand_size, out_size, { 1,1 }).bias(false)),
		bn3(torch::nn::BatchNorm2dOptions(out_size))
	{
		if (stride == 1 && in_size != out_size) {
			shortcut = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size, out_size, { 1,1 }).bias(false)), torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_size)));
		}
		

		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("nolinear1", nolinear1);
		register_module("nolinear2", nolinear2);
		register_module("se", se);
		register_module("shortcut", shortcut);
	}
	torch::Tensor forward(torch::Tensor x) {
		auto y = nolinear1->forward(bn1->forward(conv1->forward(x)));
		y = nolinear2->forward(bn2->forward(conv2->forward(y)));
		y = bn3->forward(conv3->forward(y));
		y = se->forward(y);
		if (stride_ == 1 && in_size_ != out_size_) {
			y = y + shortcut->forward(x);
		}
		else if (stride_ == 1 && in_size_ == out_size_) {
			y = y + x;
		}
		return y;
	}
	int stride_;
	int in_size_;
	int out_size_;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::ReLU nolinear1;
	torch::nn::ReLU nolinear2;
	torch::nn::BatchNorm2d bn1;
	torch::nn::BatchNorm2d bn2;
	torch::nn::BatchNorm2d bn3;
	SeModule se;
	torch::nn::Sequential shortcut;
};



struct BlockReLUNullModule : torch::nn::Module {
	BlockReLUNullModule(int kernel_size, int in_size, int expand_size, int out_size, torch::nn::ReLU nolinear, int stride) :

		stride_(stride),
		in_size_(in_size),
		out_size_(out_size),
		conv1(torch::nn::Conv2dOptions(in_size, expand_size, { 1,1 }).bias(false)),
		bn1(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear1(nolinear),
		conv2(torch::nn::Conv2dOptions(expand_size, expand_size, { kernel_size,kernel_size }).stride({ stride,stride }).padding({ kernel_size / 2,kernel_size / 2 }).groups(expand_size).bias(false)),
		bn2(torch::nn::BatchNorm2dOptions(expand_size)),
		nolinear2(nolinear),
		conv3(torch::nn::Conv2dOptions(expand_size, out_size, { 1,1 }).bias(false)),
		bn3(torch::nn::BatchNorm2dOptions(out_size))
		
	{
		
		if (stride == 1 && in_size != out_size) {
			shortcut = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_size, out_size, { 1,1 }).bias(false)), torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_size)));
		}
		
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("nolinear1", nolinear1);
		register_module("nolinear2", nolinear2);
		register_module("shortcut", shortcut);
	}
	torch::Tensor forward(torch::Tensor x) {
		auto y = nolinear1->forward(bn1->forward(conv1->forward(x)));
		y = nolinear2->forward(bn2->forward(conv2->forward(y)));
		y = bn3->forward(conv3->forward(y));
		if (stride_ == 1 && in_size_ != out_size_) {
			y = y + shortcut->forward(x);
		}
		else if (stride_ == 1 && in_size_ == out_size_) {
			y = y + x;
		}
		
		return y;
	}
	int stride_;
	int in_size_;
	int out_size_;
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::ReLU nolinear1;
	torch::nn::ReLU nolinear2;
	torch::nn::BatchNorm2d bn1;
	torch::nn::BatchNorm2d bn2;
	torch::nn::BatchNorm2d bn3;
	torch::nn::Sequential shortcut;
};

struct MobileNetV3 : torch::nn::Module {
	MobileNetV3(int num_classes = 1000)
		:
		conv1(torch::nn::Conv2dOptions(3, 16, { 3,3 }).stride({ 2,2 }).padding({ 1,1 }).bias(false)),
		bn1(torch::nn::BatchNorm2dOptions(16)),
		hs1(),
		bneck(torch::nn::Sequential(
			BlockReLUModule(3, 16, 16, 16, torch::nn::ReLU(torch::nn::functional::ReLUFuncOptions(true)), SeModule(16), 2),
			BlockReLUNullModule(3, 16, 72, 24, torch::nn::ReLU(torch::nn::functional::ReLUFuncOptions(true)), 2),
			BlockReLUNullModule(3, 24, 88, 24, torch::nn::ReLU(torch::nn::functional::ReLUFuncOptions(true)),  1),
			BlockHSwishModule(5, 24, 96, 40, HSwish(), SeModule(40), 2),
			BlockHSwishModule(5, 40, 240, 40, HSwish(), SeModule(40), 1),
			BlockHSwishModule(5, 40, 240, 40, HSwish(), SeModule(40), 1),
			BlockHSwishModule(5, 40, 120, 48, HSwish(), SeModule(48), 1),
			BlockHSwishModule(5, 48, 144, 48, HSwish(), SeModule(48), 1),
			BlockHSwishModule(5, 48, 288, 96, HSwish(), SeModule(96), 2),
			BlockHSwishModule(5, 96, 576, 96, HSwish(), SeModule(96), 1),
			BlockHSwishModule(5, 96, 576, 96, HSwish(), SeModule(96), 1)
		)),
		conv2(torch::nn::Conv2dOptions(96, 576, { 1,1 }).bias(false)),
		bn2(torch::nn::BatchNorm2dOptions(576)),
		hs2(),
		linear3(torch::nn::LinearOptions(576,1280)),
		bn3(torch::nn::BatchNorm1dOptions(1280)),
		hs3(),
		linear4(torch::nn::LinearOptions(1280, num_classes))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		
		register_module("hs1", hs1);
		register_module("hs2", hs2);
		register_module("hs3", hs3);

		register_module("linear3", linear3);
		register_module("linear4", linear4);
		register_module("bneck", bneck);
	}
	torch::Tensor forward(torch::Tensor x) {
		
		auto out = hs1->forward(bn1->forward(conv1->forward(x)));
		
		out = bneck->forward(out);
		
		out = hs2->forward(bn2->forward(conv2->forward(out)));
		
		out = torch::avg_pool2d(out, { 7,7 });
		
		out = out.view({out.size(0),-1});
		
		out = hs3->forward(bn3->forward(linear3->forward(out)));
		
		out = linear4->forward(out);
		
		return torch::log_softmax( out,1);
	}
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bn1;
	torch::nn::BatchNorm2d bn2;
	torch::nn::BatchNorm1d bn3;
	torch::nn::Linear linear3;
	torch::nn::Linear linear4;
	HSwish hs1;
	HSwish hs2;
	HSwish hs3;
	torch::nn::Sequential bneck;
};


class SimpleDataset : public  torch::data::Dataset<SimpleDataset>
{
private:
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> labels_;
public:
	explicit SimpleDataset(std::string& file_names,int size) :labels_(ReadLabels(file_names,size)) {};
	torch::data::Example<> get(size_t index) override
	{
		torch::Tensor line = std::get<0>(labels_[index]);
		torch::Tensor label = std::get<1>(labels_[index]);
		return { line,label };
	};
	torch::optional<size_t> size() const override
	{
		return labels_.size();
	};
};
