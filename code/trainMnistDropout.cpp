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



#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>



#include <limits.h>

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




struct NetImpl : torch::nn::Module {
  NetImpl()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),          //in_channels,out_channels,kernel_size,stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1_drop", conv1_drop);
    register_module("conv1", conv1);
    register_module("conv2_drop", conv2_drop);
    register_module("conv2", conv2);

    register_module("fc1_drop", fc1_drop);
    register_module("fc1", fc1);
    register_module("fc2_drop", fc2_drop);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    

    

    x = torch::relu(torch::max_pool2d(conv1_drop->forward(conv1->forward(x)), 2));
    x = torch::relu(torch::max_pool2d(conv1_drop->forward(conv2->forward(x)), 2));

    //x = torch::relu(conv1_drop->forward(torch::max_pool2d((conv1->forward(x)), 2)));
    //x = torch::relu(conv2_drop->forward(torch::max_pool2d((conv2->forward(x)), 2)));

    x = x.view({-1, 320});  
   // x = torch::dropout(x, /*p=*/0.8, /*training=*/is_training());
    
    //x = torch::relu(fc1->forward(x));                                     // in_features - size of each input sample
                                                                          // out_features - size of each output sample
                                                                          // bias - is set to false , the layer will learn an additive bias. Defaut: True

    x = feature_dropout(x, 0,is_training());

    x = torch::relu((fc1->forward(x))); 
    
    //x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = feature_dropout(x, 0,is_training());

    x = (fc2->forward(x));

    
    return torch::log_softmax(x, /*dim=*/1);
  }




  torch::nn::FeatureDropout conv1_drop;
  torch::nn::Conv2d conv1;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Conv2d conv2;

  torch::nn::FeatureDropout fc1_drop;
  torch::nn::Linear fc1;
  torch::nn::FeatureDropout fc2_drop;
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
    optimizer.zero_grad();

    
    auto output = model->forward(data).contiguous();
    auto loss = torch::nll_loss(output, targets);

    loss.backward();

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
   

  }


  test_loss /= dataset_size;

  /*std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n ",
      test_loss,
      static_cast<double>(correct) / dataset_size);*/

    std::printf(
      "\nAccuracy: %.6f\n",
      static_cast<double>(correct) / dataset_size);
}













  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// NEW DROPOUT ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////


//#define SIZE 5
//#define ODDS 3    //the odds of injecting an error


double flip_bit(double value, int bit_nr) {


    long *ptr_to_value = (long*) &value;   // access the 64bits of the double value

    *ptr_to_value ^= (1 << bit_nr);        // apply the XOR mask to flip one bit

   if(value != value)  {		//prevents -NaN
      flip_bit(value, bit_nr);

    }else {
      return value;                          // return the new double value
    }
}

double err(double value, double p) {

double value_after= 0;
long bit = 0;
	
    double r = ((double) rand() / (RAND_MAX));
    
    if(r <= p){


       do{ bit = rand() % 64;}while(bit==63 || bit==31);            // select one random bit to be flipped (without bit number 63 and 31)

      value_after = flip_bit(value, bit);

      return value_after; 						// return the value with a bit flipped
    }
    else {
        return value;                      // return the value unchanged
    }
}

int createRand(int dimension){
  int value = rand() % dimension;

  return value;

}







namespace at { namespace native {

namespace {

template<bool inplace>
using Ctype = typename std::conditional<inplace, Tensor&, Tensor>::type;

Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (int64_t i = 2; i < input.dim(); ++i)
    sizes.push_back(1);
  return at::empty(sizes, input.options());
}

bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return input.is_cuda() && p > 0 && p < 1 && input.numel() > 0;
}

// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {



TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
 //se p for igual 0 n/ é alterado qualquer valor
  if (p == 0 || !train || input.numel() == 0) {
    std::cout << input.type() << std::endl;
    return input;
  }else{     

      auto input_sizes = input.sizes();


      int size_dim1 = input_sizes[0];
      int size_dim2 = input_sizes[1];
      int size_dim3 = input_sizes[2];
      int size_dim4 = input_sizes[3];


      if(input.dim() == 2){

        for(int i = 0; i < (size_dim1 / 2); i++){

          int v1 = createRand(size_dim1);

          for(int j = 0; j < (size_dim2/ 2); j++ ){

            int v2 = createRand(size_dim2);

            double aux = 0;
            aux = err(input[v1][v2].template item<double>(),p);
            input[v1][v2] = aux;

          }

        }

      }

      else if(input.dim() == 4){

        for(int i = 0; i < (size_dim1 / 3); i++){

          int v1 = createRand(size_dim1);

          for(int j = 0; j < (size_dim2 / 3); j++){

            int v2 = createRand(size_dim2);
          
            for(int k = 0; k < (size_dim3 / 3); k++){

              long v3 = createRand(size_dim3);
          
              for(int h = 0; h < (size_dim4 / 3); h++){

                long v4 = createRand(size_dim4);

                double aux = 0; 
                aux = err(input[v1][v2][v3][v4].template item<double>(),p);
                input[v1][v2][v3][v4] = aux;

              }

            }

          }

        }
      }
      else{
        std::cout << "DIMENSION 1 -> " << size_dim1 << "DIMENSION 2 -> " << size_dim2 << "DIMENSION 3 -> " << size_dim3 << "DIMENSION 4 -> " << size_dim4 << std::endl;
          return input;
      }
      
  
      return input;
    }
}









  //Nunca vão existir bit-flips
  //if (p == 0 || !train || input.numel() == 0) {
  //  return input;
  
  //}


  //Vão sempre existir bit-flips
  //if (p == 1) {
    //return multiply<inplace>(input, at::zeros({}, input.options()));

  
      
  //    std::cout << input.data.cpu().numpy()   << std::endl;
    
  //}


//

/*
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OLD DROPOUT ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "probability -> "<< p << std::endl;
  
  //TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
     
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only


 

  // IF IT IS FEATURE_DROPOUT IT ENTERS IN THE MAKE_FEATURE_NOISE ELSE IT ENTERS IN THE EMPTY_LIKE
  // make_feature_noise -> ??????????????
  // at::empty_like(input) -> Returns an uninitialized tensor with the same size as input. torch.empty_like(input) is equivalent to torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).


  //auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input);     //CHANGED MEMORY FORMAT


  
  //Creates a Bernoulli distribution parameterized by probs or logits (but not both).
  //Samples are binary (0 or 1). They take the value 1 with probability p and 0 with probability 1 - p.
  noise.bernoulli_(1 - p);

  //ALPHA DROPOUT NOT USED 
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);

  // FEATURE_DROPOUT  
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  }
  //FEATURE DROPOUT 
  else {
    return multiply<inplace>(input, noise).add_(b);
  }

  

}
*/
#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

} // anomymous namepsace

Tensor dropout(const Tensor& input, double p, bool train) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (train && is_fused_kernel_acceptable(input, p)) {
      return std::get<0>(at::_fused_dropout(input, 1 - p));
    }
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  return result;
}

Tensor& dropout_(Tensor& input, double p, bool train) {
  return _dropout<true>(input, 0.2, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {

  //std::cout << "Feature_Dropout | p-> " << p <<  std::endl;
  return _feature_dropout<false>(input, 0.2, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
   std::cout << "&Feature_Dropout | p-> " << p <<  std::endl;
  return _feature_dropout<true>(input, 0.2, train);
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}

}} // namespace at::native






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
  //summary(model, (1,28,28));

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
  //std::cout << "DEBUG: accessfile() called by process " << ::getpid() << " (parent: " << ::getppid() << ")" << std::endl;
  std::cout << "train_dataset_size" << train_dataset_size  << std::endl;

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }

  //torch::save(model, "model10EpochsDropout.pt");  STIMULATED DROPOUT 50% -> 0.878400

  //torch::save(model, "modelStimDropout80.pt");  //STIMULATED DROPOUT 80% -> 0.877400

  torch::save(model, "modelStimDropout20.pt");  //STIMULATED DROPOUT 20% ->0.880200
	

   std::cout << "DEBUG: Train ended" << std::endl;


}
