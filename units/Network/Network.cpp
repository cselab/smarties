#include <gtest/gtest.h>

#include "smarties/Core/StateAction.h"
#include "smarties/Network/Builder.h"
#include "smarties/Network/Conv2Dfactory.h"

#include <string.h> // memcpy
#include <iostream>
#include <functional> // function
#include <algorithm>

#include <omp.h>
#include <mpi.h>

std::unique_ptr<smarties::ExecutionInfo> info;

void checkGrads(smarties::Network & NET)
{
  const unsigned seq_len = 5;
  const double incr = std::sqrt(std::numeric_limits<float>::epsilon());
  const double tol  = std::sqrt(std::numeric_limits<float>::epsilon());;
  //printf("Checking grads with increment %e and tolerance %e\n", incr,tol);

  std::vector<std::unique_ptr<smarties::Activation>> timeSeries;
  std::shared_ptr<smarties::Parameters>     grad = NET.allocParameters();
  std::shared_ptr<smarties::Parameters> backGrad = NET.allocParameters();
  backGrad->clear();
  std::shared_ptr<smarties::Parameters> diffGrad = NET.allocParameters();
  diffGrad->clear();
  std::shared_ptr<smarties::Parameters>  errGrad = NET.allocParameters();
  errGrad->clear();

  std::random_device rd;
  std::mt19937 gen(rd());
  for(unsigned t=0; t<seq_len; ++t)
  for(unsigned o=0; o<NET.nOutputs; ++o)
  {
    std::vector<std::vector<double>> inputs(seq_len,
        std::vector<double>(NET.nInputs,0));
    std::normal_distribution<double> dis_inp(0, 1);
    for(unsigned i=0; i<seq_len; ++i)
      for(unsigned j=0; j<NET.nInputs; ++j) inputs[i][j] = dis_inp(gen);

    NET.allocTimeSeries(timeSeries, seq_len);
    for (unsigned k=0; k<seq_len; ++k) {
      NET.forward(inputs[k], timeSeries, k);
      std::vector<double> errs(NET.nOutputs, 0);
      if(k==t) {
        errs[o] = -1;
        timeSeries[k]->addOutputDelta(errs);
      }
    }

    grad->clear();
    NET.backProp(timeSeries, t+1, grad.get());

    for (unsigned w=0; w<NET.weights->nParams; ++w) {
      double diff = 0;
      const auto copy = NET.weights->params[w];
      //1
      NET.weights->params[w] = copy + incr;
      for (unsigned k=0; k<seq_len; ++k) {
        const std::vector<double> ret = NET.forward(inputs[k], timeSeries, k);
        if(k==t) diff = -ret[o]/(2*incr);
      }
      //2
      NET.weights->params[w] = copy - incr;
      for (unsigned k=0; k<seq_len; ++k) {
        const std::vector<double> ret = NET.forward(inputs[k], timeSeries, k);
        if(k==t) diff += ret[o]/(2*incr);
      }
      //0
      NET.weights->params[w] = copy;

      double err = std::fabs(grad->params[w] - diff);
      if ( err > errGrad->params[w] ) {
        backGrad->params[w] = grad->params[w];
        diffGrad->params[w] = diff;
        errGrad->params[w] = err;
      }
    }
  }

  long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0, sum3 = 0, sumsq3 = 0;
  for (unsigned w=0; w<NET.weights->nParams; ++w) {
    double scale = std::max({(double) std::fabs(backGrad->params[w]),
                             (double) std::fabs(diffGrad->params[w]),
                             (double) 1.0});
    if(errGrad->params[w]>tol)
      printf("%u err:%f, grad:%f, diff:%f, param:%f\n",
              (unsigned) w, errGrad->params[w], backGrad->params[w],
              diffGrad->params[w], NET.weights->params[w]);
    ASSERT_LT(errGrad->params[w]/scale, tol);

    sum1 += std::fabs(backGrad->params[w]);
    sum2 += std::fabs(diffGrad->params[w]);
    sum3 += std::fabs(errGrad->params[w]);
    sumsq1 += backGrad->params[w] * backGrad->params[w];
    sumsq2 += diffGrad->params[w] * diffGrad->params[w];
    sumsq3 +=  errGrad->params[w] *  errGrad->params[w];
  }

  const long double NW = NET.weights->nParams;
  const auto avg1 = sum1/NW, avg2 = sum2/NW, avg3 = sum3/NW;
  const auto std1 = std::sqrt((sumsq1-sum1*avg1)/NW);
  const auto std2 = std::sqrt((sumsq2-sum2*avg2)/NW);
  const auto std3 = std::sqrt((sumsq3-sum3*avg3)/NW);
  printf("<|grad|>:%Le (std:%Le) <|diff|>:%Le (std:%Le) <|err|>::%Le (std:%Le)\n",
    avg1, std1, avg2, std2, avg3, std3);
}

TEST (Core, Network)
{
  smarties::MDPdescriptor MDP;
  MDP.dimState = 1;
  MDP.dimAction = 2;              //2 action components
  const auto sync = [](void* buffer, size_t size) {}; //no op
  MDP.synchronize(sync);
  smarties::HyperParameters HP(MDP.dimState,  MDP.dimAction);

  {
    smarties::Builder network_build(HP, * info.get());
    network_build.addInput(9);
    network_build.addLayer(8, "Tanh");
    network_build.addLayer(1, "Linear", true);
    network_build.build();
    checkGrads(* network_build.net.get() );
  }
  {
    smarties::Builder network_build(HP, * info.get());
    network_build.addInput(9);
    network_build.addLayer(8, "Tanh", false, "GRU");
    network_build.addLayer(1, "Linear", true);
    network_build.build();
    checkGrads(* network_build.net.get() );
  }
  {
    smarties::Builder network_build(HP, * info.get());
    network_build.addInput(9);
    network_build.addLayer(8, "Tanh", false, "LSTM");
    network_build.addLayer(1, "Linear", true);
    network_build.build();
    checkGrads(* network_build.net.get() );
  }
  {
    smarties::Builder network_build(HP, * info.get());
    static constexpr int inp_x = 6, inp_y = 3, inp_n = 2;
    network_build.addInput(inp_x * inp_y * inp_n);
    static constexpr int filter_x = 3,  filter_y = 2, filter_n = 2;
    static constexpr int stride_x = 1,  stride_y = 1, pad_x = 0, pad_y = 0;
    static constexpr int out_x = (inp_x - filter_x + 2*pad_x)/stride_x + 1;
    static constexpr int out_y = (inp_y - filter_y + 2*pad_y)/stride_y + 1;
    smarties::makeConv2D<inp_x, inp_y, inp_n, filter_x, filter_y, filter_n,
               stride_x, stride_y, pad_x, pad_y, out_x, out_y>(network_build);
    network_build.addLayer(1, "Linear", true);
    network_build.build();
    checkGrads(* network_build.net.get() );
  }
  {
    smarties::Builder network_build(HP, * info.get());
    static constexpr int inp_x = 8, inp_y = 10, inp_n = 2;
    network_build.addInput(inp_x * inp_y * inp_n);
    static constexpr int filter_x = 4,  filter_y = 5, filter_n = 3;
    static constexpr int stride_x = 2,  stride_y = 3, pad_x = 1, pad_y = 1;
    static constexpr int out_x = (inp_x - filter_x + 2*pad_x)/stride_x + 1;
    static constexpr int out_y = (inp_y - filter_y + 2*pad_y)/stride_y + 1;
    smarties::makeConv2D<inp_x, inp_y, inp_n, filter_x, filter_y, filter_n,
               stride_x, stride_y, pad_x, pad_y, out_x, out_y>(network_build);
    network_build.addLayer(1, "Linear", true);
    network_build.build();
    checkGrads(* network_build.net.get() );
  }
}

int main(int argc, char **argv)
{
  info = std::make_unique<smarties::ExecutionInfo>(argc, argv);
  info->initialze();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

