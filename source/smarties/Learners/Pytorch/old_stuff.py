


    # def clipImpWeight(self, impWeight):
    #     impWeight = [7.0 if(impWeight_i>7.0) else (-7.0 if impWeight_i < -7.0 else impWeight_i) for impWeight_i in impWeight]
    #     return impWeight

# USEFULL COMMANDS:
# import pybind11
# if(PRINT_OUTPUT): print(pybind11.__file__)
# if(PRINT_OUTPUT): print(pybind11.__version__)



    # def evaluateImportanceWeight(self, sampled_action, policy_mean, policy_sigma, behavior_mean, mu_sigma):
    #     polLogProb = self.evalLogProbability(sampled_action, policy_mean, policy_sigma)
    #     behaviorLogProb = self.evalLogProbability(sampled_action, behavior_mean, mu_sigma)
    #     impWeight = polLogProb - behaviorLogProb
    #     impWeight = self.clipImpWeight(impWeight)
    #     return np.exp(impWeight)

    # def evalLogProbability(self, var, mean, sigma):
    #     p=np.zeros(np.shape(mean)[0])
    #     for i in range(np.shape(mean)[1]):
    #         precision = 1.0 / np.power(sigma[:,i], 2.0)
    #         p -= precision * np.power((var[:,i]-mean[:,i]), 2.0)
    #         p += np.log( precision / 2.0 / math.pi )
    #     return 0.5 * p





    # def evaluateKLDivergence(self, action, policy_mean, policy_sigma, behavior_mean, behavior_sigma):
    #     policy_mean=np.random.rand(36,2)
    #     behavior_mean=np.random.rand(36,2)
    #     policy_sigma=np.random.rand(36,2)
    #     behavior_sigma=np.random.rand(36,2)
    #     det_sigma_policy = np.prod(policy_sigma, axis=1)
    #     det_sigma_behavior = np.prod(behavior_sigma, axis=1)
    #     N = np.shape(policy_mean)[1]

    #     # Building the (policy_mean-behavior_mean)' * policy_sigma^{-1} * (policy_mean-behavior_mean)
    #     temp = policy_mean - behavior_mean
    #     temp = temp * temp
    #     temp = temp / policy_sigma
    #     temp = np.sum(temp, axis=1)

    #     kldiv = np.log( det_sigma_policy / det_sigma_behavior ) + np.sum(behavior_sigma / policy_sigma, axis=1) + temp - N 

    #     kldiv = 0.5 * kldiv
    #     return kldiv


    # def forwardBatchId(self, input_dict):
    #     PRINT_OUTPUT=False
    #     vectorMiniBatch = input_dict["vectorMiniBatch"]
    #     bID = input_dict["bID"]
    #     t = input_dict["t"]
    #     fval = input_dict["fval"]
    #     input_ = np.array(vectorMiniBatch[0].S)
    #     E, T, D = np.shape(input_)
    #     input_ = input_[bID, :t, :]
    #     input_batch = input_[:, -self.T:, :]
    #     E, T, D = np.shape(input_batch)
    #     output, sigma = self.forwardVector(input_batch)
    #     output_det = output.detach().numpy()
    #     sigma_det = sigma.detach().numpy()
    #     # COMPUTING VALUE FUNCTION
    #     fval = output_det[0,0]
    #     return 0

    
# OLD FUNCTIONS

    # def setPytorchWeightsFromCppWeights2(self, input_dict):
    #     print(input_dict)
    #     print(input_dict["pytorchWeights"][0])

    #     weights_array = input_dict["pytorchWeights"][0].getWeights()

    #     sizes = input_dict["pytorchWeights"][0].getSizes()
    #     total_size = input_dict["pytorchWeights"][0].getTotalSizes()
    #     print(weights_array)
    #     print(sizes)
    #     print(total_size)
    #     i=0
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             temp = np.array(weights_array[i])
    #             print("SIZE:")
    #             print(sizes[i])
    #             print(np.shape(param.data))

    #             size = np.array(sizes[i])
    #             temp = np.reshape(temp, size)
    #             param.data = torch.DoubleTensor(temp)
    #             i+=1
    #     print("WEIGHTS SUCCESSFULLY SEND TO PYTORCH !")
    #     return 0



    # def getWeightsReturn(self, dummy):

    #     # weightsVector = input_dict["pytorchWeights"][0]
    #     # weightsVector = input_dict
    #     # print(weightsVector)

    #     # array1 = np.array([1.2,4.2,3.3,4.5,6.7,8.8])
    #     # array2 = np.array([1.2,4.5,6.7,8.8])
    #     # array3 = np.array([1.2,4.2,3.3])
    #     # weightsVector.saveWeights([array1,array2,array3])
    #     # array2[2]=1000.0
    #     # weightsVector.printWeights()
    #     print("PYTHON-NET: Getting weights.")
    #     weights_array = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             # print(name)
    #             data = param.data
    #             print(np.shape(data))
    #             weights_array.append(data)

    #     # weightsVector.saveWeights(weights_array)
    #     # print("PYTHON-NET: Weights have been sent.")
    #     # self.setWeights(weights_array)
    #     return weights_array

    # def getWeightsFromPytorchReference(self):
    #     weights_array = []
    #     for name, param in self.named_parameters():
    #         print("name", name)
    #         if param.requires_grad:
    #             data = param.data
    #             weights_array.append(data)

    #     # print("getWeightsFromPytorchReference.")
    #     return weights_array

# def getWeightPointers(self):
#     # print(self.parameters)
#     # print(self.parameters())
#     # for param in self.parameters():
#         # print(param)

#     param_array = []
#     for name, param in self.named_parameters():
#         if param.requires_grad:
#             print(name)
#             print(param.data)
#             # print(param.data_ptr)
#             data_ptr = param.data_ptr()
#             print(data_ptr)
#             # Return the size of an object in bytes
#             print(np.shape(param.data))
#             size = sys.getsizeof(param.data.storage())
#             print(size)

#     a=[1.123,2.23423,3.2]
#     size = sys.getsizeof(a)
#     print(size)
#     return 0



    # def setWeights(self, input_dict):
    #     print(input_dict)
    #     print(input_dict["pytorchWeights"][0])

    #     weights_array = input_dict["pytorchWeights"][0].getWeights()

    #     sizes = input_dict["pytorchWeights"][0].getSizes()
    #     total_size = input_dict["pytorchWeights"][0].getTotalSizes()
    #     print(weights_array)
    #     print(sizes)
    #     print(total_size)
    #     i=0
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             temp = np.array(weights_array[i])
    #             print("SIZE:")
    #             print(sizes[i])
    #             print(np.shape(param.data))

    #             size = np.array(sizes[i])
    #             temp = np.reshape(temp, size)
    #             param.data = torch.DoubleTensor(temp)
    #             i+=1
    #     print("WEIGHTS SUCCESSFULLY SET !")
    #     return 0


    # def getWeightsFromPytorch(self, input_dict):

    #     weightsVector = input_dict["pytorchWeights"][0]
    #     # weightsVector = input_dict
    #     print(weightsVector)
    #     # array1 = np.array([1.2,4.2,3.3,4.5,6.7,8.8])
    #     # array2 = np.array([1.2,4.5,6.7,8.8])
    #     # array3 = np.array([1.2,4.2,3.3])
    #     # weightsVector.saveWeights([array1,array2,array3])
    #     # array2[2]=1000.0
    #     # weightsVector.printWeights()
    #     print("PYTHON-NET: Getting weights.")
    #     weights_array = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    #             data = param.data
    #             print(np.shape(data))
    #             weights_array.append(data)

    #     weightsVector.saveWeights(weights_array)
    #     print("PYTHON-NET: Weights have been sent.")
    #     # self.setWeights(weights_array)
    #     return 0


    # def transformMiniBatchTEST(self, input_, idx, BBTT_steps):
    #     # As the input_ contains multiple time-steps of some episode and we are training on a specific time-step sampled at time idx, we need to shift the input_ data
    #     # print("INPUT_ SHAPE:")
    #     # print(np.shape(input_))
    #     input_batch = []
    #     for i in range(len(input_)):
    #         # item has shape [T x D]
    #         item = input_[i]
    #         # print("item_ SHAPE:")
    #         # print(np.shape(item))
    #         # Taking only the relevant time-steps
    #         start_idx = idx[i] - BBTT_steps
    #         end_idx = idx[i] + 1
    #         if(start_idx<0):
    #             D = len(item[0])
    #             # Filling the batch with zeros
    #             temp = np.zeros((BBTT_steps, D))
    #             temp[-end_idx:] = item[:end_idx]
    #             input_batch.append(temp)
    #         else:
    #             input_batch.append(item[start_idx:end_idx])
    #     input_batch = np.array(input_batch)
    #     return input_batch



# Timer unit: 1e-06 s

# Total time: 0.011842 s
# File: /home/pvlachas/smarties/source/Learners/Pytorch/net_modules.py
# Function: trainOnBatch_ at line 140

# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#    140                                               def trainOnBatch_(self, input_dict):
#    141                                                   # with torch.autograd.profiler.profile(use_cuda=False) as profiler_:
#    142
#    143         1         72.0     72.0      0.6          self.optimizer.zero_grad()
#    144         1          3.0      3.0      0.0          PRINT_OUTPUT=False
#    145         1          2.0      2.0      0.0          if(PRINT_OUTPUT): print("PYTHON-NET: trainOnBatch")
#    146         1          2.0      2.0      0.0          if(PRINT_OUTPUT): print("PYTHON-NET: Input to trainOnBatch:")
#    147
#    148         1          2.0      2.0      0.0          vectorMiniBatch = input_dict["vectorMiniBatch"]
#    149
#    150         1         23.0     23.0      0.2          Q_RET_cppvec = vectorMiniBatch[0].get_Q_RET(0)
#    151         1          9.0      9.0      0.1          isNextTerminal_cppvec = vectorMiniBatch[0].isNextTerminal()
#    152         1          6.0      6.0      0.1          isNextTruncated_cppvec = vectorMiniBatch[0].isNextTruncated()
#    153         1         15.0     15.0      0.1          mu_cppvec = vectorMiniBatch[0].get_MU(0)
#    154
#    155                                                   # print("mu_cppvec")
#    156                                                   # print(mu_cppvec)
#    157                                                   # print(mu_cppvec[0])
#    158                                                   # print(mu_cppvec[1])
#    159                                                   # print(mu_cppvec[2])
#    160         1        696.0    696.0      5.9          mu_tensor = torch.DoubleTensor(mu_cppvec)
#    161                                                   # print("mu_tensor")
#    162                                                   # print(mu_tensor)
#    163                                                   # print(mu_tensor.size())
#    164
#    165                                                   # print("Q_RET_cppvec")
#    166                                                   # print(Q_RET_cppvec)
#    167         1         57.0     57.0      0.5          Q_RET_tensor = torch.DoubleTensor(Q_RET_cppvec)
#    168                                                   # print("Q_RET_tensor")
#    169                                                   # print(Q_RET_tensor.size())
#    170
#    171                                                   # print("vectorMiniBatch[0].S")
#    172                                                   # print(vectorMiniBatch[0].S)
#    173                                                   # print(vectorMiniBatch[0].S[0][0])
#    174
#    175                                                   # GET THE STATE SIZE
#    176         1         19.0     19.0      0.2          max_state = len(vectorMiniBatch[0].S[0][0])
#    177                                                   # max_state = max([len(STATE) for SEQ in vectorMiniBatch[0].S for STATE in SEQ]) # MAXIMUM STATE DIM
#    178         1         98.0     98.0      0.8          max_seq = max([len(SEQ) for SEQ in vectorMiniBatch[0].S]) # MAXIMUM SEQUENCE
#    179                                                   # PADDING THE INPUT BATCH - SOME SEQUENCES DO NOT CONTAIN ALL TIME-STEPS - THROWING AWAY THE LAST ELEMENT (SAVING ONE MORE TIME-STEP NEEDED BY SOME METHODS)
#    180                                                   # IF TRAINED BY SEQUENCES REMEMBER TO CHANGE THIS BEHAVIOR!!!!
#    181         1       6362.0   6362.0     53.7          input_batch = torch.DoubleTensor([[[0.]*max_state] * (max_seq - len(SEQ)) + list(SEQ)[:-1] for SEQ in vectorMiniBatch[0].S])
#    182                                                   # print(input_batch.size())
#    183                                                   # # DEBUGGING, OUTPUT SHOULD BE THE SAME
#    184                                                   # begTimeStep = np.array(vectorMiniBatch[0].begTimeStep)
#    185         1         39.0     39.0      0.3          sampledTimeStep = np.array(vectorMiniBatch[0].sampledTimeStep)
#    186                                                   # idx = sampledTimeStep - begTimeStep
#    187                                                   # input_cppvec = np.array(vectorMiniBatch[0].S)
#    188                                                   # print(np.shape(input_cppvec))
#    189                                                   # input_batch_ = self.transformMiniBatch(input_cppvec, idx)
#    190                                                   # print(input_batch[0])
#    191                                                   # print(input_batch_[0])
#    192
#    193                                                   # isNextTerminal_tensor = torch.DoubleTensor(isNextTerminal_cppvec)
#    194                                                   # isNextTruncated_tensor = torch.DoubleTensor(isNextTruncated_cppvec)
#    195
#    196         1         11.0     11.0      0.1          E, T, D = input_batch.size()
#    197         1          3.0      3.0      0.0          if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
#    198         1          5.0      5.0      0.0          if not (T==self.BBTT_steps+1): raise ValueError("Number of time-steps does not match with BBTT_steps+1")
#    199
#    200                                                   # # LINE PROFILER
#    201                                                   # lp = LineProfiler()
#    202                                                   # lp_wrapper = lp(self.forwardVector)
#    203                                                   # lp_wrapper(input_batch)
#    204                                                   # lp.print_stats()
#    205
#    206         1        619.0    619.0      5.2          output_tensor, policy_sigma_tensor = self.forwardVector(input_batch)
#    207
#    208
#    209         1          3.0      3.0      0.0          if(PRINT_OUTPUT): print("PYTHON-NET: output = ", output)
#    210                                                   # output_tensor.register_hook(print)
#    211
#    212         1         11.0     11.0      0.1          output_np = output_tensor.detach().numpy()
#    213         1          5.0      5.0      0.0          policy_sigma_np = policy_sigma_tensor.detach().numpy()
#    214
#    215         1          3.0      3.0      0.0          action_dim = (self.output_dim-1)
#    216
#    217         1         17.0     17.0      0.1          policy_mean_tensor = output_tensor[:,1:1+action_dim]
#    218         1          6.0      6.0      0.1          policy_mean_np = output_np[:,1:1+action_dim]
#    219
#    220         1         84.0     84.0      0.7          sampled_action_np = np.random.normal(loc=policy_mean_np,scale=policy_sigma_np)
#    221         1         20.0     20.0      0.2          sampled_action_tensor = torch.DoubleTensor(sampled_action_np)
#    222
#    223         1         15.0     15.0      0.1          V_cur_tensor = output_tensor[:,0]
#    224         1          7.0      7.0      0.1          V_cur_np = V_cur_tensor.detach().numpy()
#    225
#    226         1          4.0      4.0      0.0          idx=mu_tensor.size(1)
#    227         1          3.0      3.0      0.0          if not (idx%2==0): raise ValueError("Problem with the dimensionality of the behavior policy mu.")
#    228         1         10.0     10.0      0.1          mu_mean_tensor = mu_tensor[:,:idx//2]
#    229         1          7.0      7.0      0.1          mu_sigma_tensor = mu_tensor[:,idx//2:]
#    230
#    231         1          3.0      3.0      0.0          CmaxRet = input_dict["CmaxRet"]
#    232         1          2.0      2.0      0.0          CinvRet = input_dict["CinvRet"]
#    233         1          2.0      2.0      0.0          beta = input_dict["beta"]
#    234
#    235                                                   #   # ASSUME SCALING OF ACTION TAKEN CARE OF - sampAct = map_action(unbΑct);
#    236                                                   #   pol.prepare(ACT, MU); THE ACTIONS IN THE FOLLOWING FUNCTION NEEEDS TO BE UNSCALED
#    237
#    238         1         56.0     56.0      0.5          if torch.any(mu_sigma_tensor<=0) or torch.any(policy_sigma_tensor<=0):
#    239                                                       raise ValueError("## A variance is zero during batch training, something is broken.")
#    240                                                   else:
#    241         1        402.0    402.0      3.4              rho_tensor = self.evaluateImportanceWeightTensor(sampled_action_tensor, policy_mean_tensor, policy_sigma_tensor, mu_mean_tensor, mu_sigma_tensor)
#    242         1          8.0      8.0      0.1              rho_np = rho_tensor.detach().numpy()
#    243         1         43.0     43.0      0.4              rho_np = self.clipImpWeight(rho_np)
#    244         1         20.0     20.0      0.2              rho_constant_tensor = torch.DoubleTensor(rho_np)
#    245
#    246
#    247         1        379.0    379.0      3.2          isFarPol_np = self.isFarPolicy(rho_constant_tensor, CmaxRet, CinvRet)
#    248         1         42.0     42.0      0.4          isFarPol_tensor = torch.DoubleTensor(isFarPol_np)
#    249
#    250         1         15.0     15.0      0.1          A_cur_tensor = self.getAdvantageTensor(sampled_action_tensor)
#    251         1         21.0     21.0      0.2          A_RET_tensor = Q_RET_tensor - V_cur_tensor
#    252         1         43.0     43.0      0.4          Ver_tensor = torch.min(rho_constant_tensor, torch.ones_like(rho_constant_tensor)) * (A_RET_tensor-A_cur_tensor)
#    253
#    254         1         13.0     13.0      0.1          Ver_2_tensor = torch.pow(Ver_tensor, 2.0)
#    255         1         46.0     46.0      0.4          value_loss = beta * (1.0 - isFarPol_tensor) * Ver_2_tensor
#    256         1         41.0     41.0      0.3          value_loss_mean = torch.mean(value_loss)
#    257
#    258         1         45.0     45.0      0.4          Aer_tensor = torch.min(rho_constant_tensor, CmaxRet*torch.ones_like(rho_constant_tensor)) * (A_RET_tensor-A_cur_tensor)
#    259
#    260         1         11.0     11.0      0.1          A_cur_np = A_cur_tensor.detach().numpy()
#    261         1        162.0    162.0      1.4          self.updateRetrace(vectorMiniBatch, sampledTimeStep, A_cur_np, V_cur_np, rho_np)
#    262
#    263         1        197.0    197.0      1.7          dkl_tensor = self.evaluateKLDivergenceTensor(sampled_action_tensor, policy_mean_tensor, policy_sigma_tensor, mu_mean_tensor, mu_sigma_tensor)
#    264         1         15.0     15.0      0.1          dkl_loss = (1-beta)*dkl_tensor
#    265         1         31.0     31.0      0.3          dkl_loss_mean = torch.mean(dkl_loss)
#    266
#    267         1         43.0     43.0      0.4          temp = A_RET_tensor * torch.min(CmaxRet*torch.ones_like(rho_tensor), rho_tensor)
#    268         1         22.0     22.0      0.2          policy_loss = - beta * rho_tensor * temp
#    269         1         28.0     28.0      0.2          policy_loss_mean = torch.mean(policy_loss)
#    270
#    271         1         79.0     79.0      0.7          self.loss = value_loss_mean + dkl_loss_mean + policy_loss_mean
#    272
#    273                                                   # self.loss.backward(retain_graph=True)
#    274         1       1141.0   1141.0      9.6          self.loss.backward()
#    275         1        469.0    469.0      4.0          self.optimizer.step()
#    276         1         38.0     38.0      0.3          self.optimizer.zero_grad()
#    277
#    278                                                   # print(self.loss)
#    279
#    280         1          7.0      7.0      0.1          dkl_np = dkl_tensor.detach().numpy()
#    281         1          5.0      5.0      0.0          Ver_2_np = Ver_2_tensor.detach().numpy()
#    282         1        172.0    172.0      1.5          self.setMseDklImpw(vectorMiniBatch, sampledTimeStep, Ver_2_np, dkl_np, rho_np, CmaxRet, CinvRet)
#    283
#    284                                                   # PRINTING THE PROFILER
#    285                                                   # print(profiler_)
#    286         1          3.0      3.0      0.0          return 0

# Timer unit: 1e-06 s


