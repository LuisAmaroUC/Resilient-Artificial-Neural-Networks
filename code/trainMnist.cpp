#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

//Files
#include <fstream>
#include <string>
using namespace std;

#include "/home/luisamaro/Desktop/cpp-subprocess-master/include/subprocess.hpp"

//Generate random
#include <stdlib.h>

#include <unistd.h>		// ::getpid()


// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

//HW_Fault_Injection_Path
//std::string path = "/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject_intel";


struct NetImpl : torch::nn::Module {
  NetImpl()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    //register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
    	torch::max_pool2d((conv2->forward(x)), 2));
    //    torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    //x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  //torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};
TORCH_MODULE(Net);


template <typename DataLoader>
void train(
    size_t epoch,
    Net model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
    model->train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
  	
    auto data = batch.data.to(device), targets = batch.target.to(device);

    //Set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    optimizer.zero_grad();

    auto output = model->forward(data);

    //IN THE FORWARD PHASE AUTOGRAD TAPE WILL REMEMBER ALL THE OPERATIONS IT EXECUTED, AND IN THE BACKWARD PHASE, IT WILL REPLAY OPERATIONS.
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));

    //IT ONLY UPGRADES THE GRADIENT
    loss.backward();

    //UPDATES THE WEIGHTS.    
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
     std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size,
    char* argv[]) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;

  int count = 1;
  srand (time(NULL));
  int randomNumber = rand() % 10 + 1;

  for (const auto& batch : data_loader) {
  	
  	//2091490088
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                      Reduction::Sum)
                     //at::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
   
  //  cout << count << " COUNT";
  //  cout << randomNumber << " randomNumber";
  /* 
    if(count == randomNumber){
	    std::stringstream stream;    
	  	stream << "/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject_intel"
	        << " " // don't forget a space between the path and the arguments
	       	<< std::to_string(::getpid()) // pid
	       	<< " " // don't forget a space between the path and the arguments
	       	<< argv[0]
	       	<< " " // don't forget a space between the path and the arguments
	       	<< argv[1]
	       	<< " " // don't forget a space between the path and the arguments
	       	<< "0";

	  	system(stream.str().c_str());
	}
	count++;*/
  }


  test_loss /= dataset_size;

     /*  std::stringstream stream;    
	  	stream << "/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject_intel"
	        << " " // don't forget a space between the path and the arguments
	       	<< std::to_string(::getpid()) // pid
	       	<< " " // don't forget a space between the path and the arguments
	       	<< argv[0]
	       	<< " " // don't forget a space between the path and the arguments
	       	<< argv[1]
	       	<< " " // don't forget a space between the path and the arguments
	       	<< "0";

	  	system(stream.str().c_str());*/
  /*std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n ",
      test_loss,
      static_cast<double>(correct) / dataset_size);*/

    std::printf(
      "\nAccuracy: %.6f\n",
      static_cast<double>(correct) / dataset_size);
}

auto main(int argc, char* argv[]) -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model->to(device);

                                              
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  
  //print PID do projeto
  std::cout << "DEBUG: accessfile() called by process " << ::getpid() << " (parent: " << ::getppid() << ")" << std::endl;
  
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }

  torch::save(model, "model10Epochs.pt");
	

 /* std::string path = "/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject_intel";
  std::string pid = std::to_string(::getpid());
  std::string registo = argv[0] ;
  std::string bit = argv[1];
  std::string time = argv[2];
  std::string cmdString;
  cmdString = path+ " " + pid + " " + bit + " " + time;

  cout << cmdString;

  subprocess::popen cmd(path, {}); 
  std::cout << cmd.stdout().rdbuf();

  */
  	  //Do the hardware fault injection

   //std::cout << summary(model, (1,28,28)) << std::endl;

   //Net newModel;

   
   //torch::load(newModel,"model10Epochs.pt");

  //test(model, device, *test_loader, test_dataset_size, argv);

  //cout << current_time << " seconds has passed since the begining of testing";
  //serialize::OutputArchive output_archive;
  //model->save(output_archive);
  //output_archive.save_to("mode.pt");
  //storch::save(model, 'model.pt');

	//torch::save(model.state_dict(), "model.pt");

	/*Net newModel;
  	newModel.to(device);
	torch::load(newMode,"model.pt");
	 

	test(newModel, device, *test_loader, test_dataset_size);*/

}
