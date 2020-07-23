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


//Generate random
#include <stdlib.h>

#include <unistd.h>		// ::getpid()

#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <signal.h>
#include <stdlib.h>
#include <ucontext.h>
// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


struct NetImpl : torch::nn::Module {
  NetImpl()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),          //in_channels,out_channels,kernel_size,stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv1_drop", conv1_drop);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);

    register_module("fc1_drop", fc1_drop);
    register_module("fc1", fc1);
    register_module("fc2_drop", fc2_drop);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {

    //x = torch::relu(conv1_drop->forward(torch::max_pool2d((conv1->forward(x)), 2)));
    //x = torch::relu(conv2_drop->forward(torch::max_pool2d((conv2->forward(x)), 2)));


    x = torch::relu(torch::max_pool2d(conv1_drop->forward(conv1->forward(x)), 2));
    x = torch::relu(
      //torch::max_pool2d((conv2->forward(x)), 2));
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});  
   // x = torch::dropout(x, /*p=*/0.8, /*training=*/is_training());
    //x = torch::relu(fc1->forward(x));                                     // in_features - size of each input sample
                                                                          // out_features - size of each output sample
                                                                          // bias - is set to false , the layer will learn an additive bias. Defaut: True

    x = feature_dropout(x, 0.5,is_training());
    x = torch::relu((fc1->forward(x))); 
    
    //x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());//POR ANTES
    x = feature_dropout(x, 0.5,is_training());

    x = (fc2->forward(x));
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::FeatureDropout conv1_drop;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;

  torch::nn::FeatureDropout fc1_drop;
  torch::nn::Linear fc1;
  torch::nn::FeatureDropout fc2_drop;
  torch::nn::Linear fc2;
};
TORCH_MODULE(Net);




template <typename DataLoader>
void test(Net model, torch::Device device, DataLoader& data_loader, size_t dataset_size, char* argv[]) {


  torch::NoGradGuard no_grad;

  model->eval();

  double test_loss = 0;

  int32_t correct = 0;

  int count = 1;



    for (const auto& batch : data_loader) {
    	

        
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

     // }

      std::ofstream myfile;
      myfile.open("pathToFile/Results.txt", std::ios_base::app); // append instead of overwrite
      myfile << setprecision(10) << static_cast<double>(correct) / dataset_size << "\n";
      myfile.close();



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


/*

TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
 //se p for igual 0 n/ é alterado qualquer valor
  if (p == 0 || !train || input.numel() == 0) {
   // std::cout << "probability == 0"<< std::endl;
    return input;
  }else{
    //std::cout << "probability not zero -> "<< p << std::endl;


      //std::cout << "DIMENSIONS" << input.sizes() << std::endl;
      

      auto input_sizes = input.sizes();

      int64_t size_dim1 = input_sizes[0];
      int64_t size_dim2 = input_sizes[1];
      int64_t size_dim3 = input_sizes[2];
      int64_t size_dim4 = input_sizes[3];

      //std::cout << "DIMENSION 1 " << size_dim1 << "DIMENSION 2" << size_dim2 << "DIMENSION 3" << size_dim3 << "DIMENSION 4" << size_dim4 << std::endl;

      //NOS PODEMOS TER TENSOR COM DIFERENTES DIMENSOES (1,2,3,4), POSTO ISTO TEMOS DE COBRIR ESTAS DIFERENTES OPÇOES
      
      

    //double value =input[0][0][0][0].template item<double>();    //converter de at::Tensor -> c10::Scalar -> double

    //  std::cout << value << std::endl;
     
      if(input.dim() == 1){
        //std::cout << " 1 DIMENSION" << std::endl;
       
        for(int i = 0; i < size_dim1; i++){
            std::cout << " 1 DIMENSION" << std::endl;
            //std::cout << " Before bit-flip" << input[i] <<std::endl;
            input[i] = err(input[i].template item<double>(), p);
           // std::cout << " After bit-flip" << input[i] <<std::endl;
        }

      }else if(input.dim() == 2){
       // std::cout << " 2 DIMENSION" << std::endl;
          for(int i = 0; i < size_dim1; i++){
            for(int j = 0; j < size_dim2; j++){
              std::cout << " 2 DIMENSION" << std::endl;
               // std::cout << " Before bit-flip" << input[i][j] <<std::endl;
                input[i][j] = err(input[i][j].template item<double>(), p);
               // std::cout << " After bit-flip" << input[i][j] <<std::endl;
            }
          }

      }else if(input.dim() == 3){
        //std::cout << " 3 DIMENSION" << std::endl;
          for(int i = 0; i < size_dim1; i++){
            for(int j = 0; j < size_dim2; j++){
              for(int k = 0; k < size_dim3; k++){
                std::cout << " 3 DIMENSION" << std::endl;
               // std::cout << " Before bit-flip" << input[i][j][k] <<std::endl;
                input[i][j][k] = err(input[i][j][k].template item<double>(), p);
               // std::cout << " After bit-flip" << input[i][j][k] <<std::endl;
              }
            }
          }

      }else if(input.dim() == 4){
        //std::cout << " 4 DIMENSION" << std::endl;

          for(int i = 0; i < size_dim1; i++){
            for(int j = 0; j < size_dim2; j++){
              for(int k = 0; k < size_dim3; k++){
                for(int l = 0;l < size_dim4; l++){
                  std::cout << " 4 DIMENSION" << std::endl;
                 // std::cout << " Before bit-flip" << input[i][j][k][l] <<std::endl;
                  input[i][j][k][l] = err(input[i][j][k][l].template item<double>(),p);

                 // std::cout << " After bit-flip" << input[i][j][k][l] <<std::endl;
                }
              }
            }
          }
      }else{
        std::cout << " + 4 DIMENSION" << std::endl;
          return input;
      }

  
      return input;
    }
}


*/






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


  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// OLD DROPOUT ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //std::cout << "probability -> "<< p << std::endl;
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
     //std::cout << "probability == 0"<< std::endl;
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
  return _dropout<true>(input, 0.5, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {

  //std::cout << "Feature_Dropout | p-> " << p <<  std::endl;

  return _feature_dropout<false>(input, 0.5, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
   std::cout << "&Feature_Dropout | p-> " << p <<  std::endl;
  return _feature_dropout<true>(input, 0.5, train);
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
    //std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;

 
  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  
  //print PID do projeto
  std::cout << "DEBUG: accessfile() called by process " << ::getpid();
  
  //torch::load(model,"model10EpochsDropout.pt");  //STIMULATED DROPOUT 50% -> 0.878400

  //torch::load(model, "modelStimDropout80.pt");  //STIMULATED DROPOUT 80% -> 0.877400

  torch::load(model, "modelStimDropout20.pt");  //STIMULATED DROPOUT 20% -> 0.880200
  //torch::load(model,"model.pt");           //Trained with 1 epoch
  //savePid = ::getpid();
  
  //pid_t pid = fork();
  //if(pid == 0) HW_Fault_Injection(argv);
  //else if(pid > 0){
  int i = 0;
  while(true) test(model, device, *test_loader, test_dataset_size, argv);



  //}else {
  //    printf("Fork Failed");
  //}


}

