import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

PRINT_OUTPUT=True

# USEFULL COMMANDS:
# import pybind11
# if(PRINT_OUTPUT): print(pybind11.__file__)
# if(PRINT_OUTPUT): print(pybind11.__version__)

class LSTM(nn.Module):
    def __init__(self, input_dim, num_hidden, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim, num_hidden, num_layers, batch_first=True)
        self.outputLayer = nn.Linear(num_hidden, self.output_dim)

    def forward(self, x, h=None, c=None):
        if (h is None) or (c is None):
            batch_size = x.size()[0]
            h, c = self.initializeHiddenState(batch_size, self.num_hidden)
        out, states = self.rnn(x, (h, c))
        # Next line can be (un)commented depending on whether you need all outputs from the previous processed time-steps or not
        out = out[:, -1, :]
        out = self.outputLayer(out)
        return out

    def initializeHiddenState(self, batch_size, num_hidden):
        hx = Variable(torch.zeros(self.num_layers, batch_size, num_hidden))
        cx = Variable(torch.zeros(self.num_layers, batch_size, num_hidden))
        return hx, cx


class NET(nn.Module):
    def __init__(self, net_type, BBTT_steps, input_dim, nnLayerSizes, output_dim, sigma_dim=0):
        super(NET, self).__init__()
        # Input has the form [K x BBTT_steps+1 x input_dim]
        # where K is the batch-size
        # BBTT_steps is the number of BBTT time-steps (POMDP)
        # input_dim is the dimensionality (e.g. state)
        self.net_type = net_type
        self.BBTT_steps = BBTT_steps
        self.input_dim = input_dim
        self.nnLayerSizes = list(nnLayerSizes)
        self.output_dim = output_dim
        self.sigma_dim = sigma_dim

        # Adding the input and output layer
        self.nnLayerSizes.insert(0, self.input_dim)
        self.nnLayerSizes.append(output_dim)

        if self.net_type == "LSTM":
            self.lstmLayerSizes = self.nnLayerSizes[1:-1]
            self.num_lstm_layers = len(self.lstmLayerSizes)
            if not (len(self.lstmLayerSizes)!=0): raise ValueError("Problem with the number of LSTM layers.")
            self.num_hidden = self.lstmLayerSizes[0]
            if not (np.all(np.array(self.lstmLayerSizes)==self.num_hidden)): raise ValueError("Not all stacked LSTM layers have the same number of hidden units.")
            self.LSTM = LSTM(self.input_dim, self.num_hidden, self.num_lstm_layers, self.output_dim)
        elif self.net_type == "FFNN":
            # Define affine operations: y = Wx + b
            self.fcLayers = []
            for ln in range(len(self.nnLayerSizes)-1):
                self.fcLayers.append(nn.Linear(self.nnLayerSizes[ln], self.nnLayerSizes[ln+1]))
        else:
            raise ValueError("net_type not recognised.")


        # Define the parameter layer for sigma
        sigma_init = 1.85
        if(sigma_dim>0):
            self.sigma_weights = Variable(torch.Tensor(sigma_init*np.array(sigma_dim*[1.0])), requires_grad=True)

    def forwardBatchId(self, input_dict):
        if(PRINT_OUTPUT): print("PYTHON-NET: forwardBatchId")
        if(PRINT_OUTPUT): print("PYTHON-NET: Input to forwardBatchId:")

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        bID = input_dict["bID"]
        t = input_dict["t"]
        fval = input_dict["fval"]

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
        input_ = input_[bID, :t, :]

        input_red = input_[:, -self.T:, :]
        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps USED x Dimensionality = ", E, T, D)

        output, sigma = self.forwardVector(input_red)
        if(PRINT_OUTPUT): print("PYTHON-NET: output = ", output)
        output_det = output.detach().numpy()
        sigma_det = sigma.detach().numpy()

        # COMPUTING VALUE FUNCTION
        fval = output_det[0,0]
        return 0


    def trainOnBatch(self, input_dict):
        # PRINT_OUTPUT=True
        if(PRINT_OUTPUT): print("PYTHON-NET: trainOnBatch")
        if(PRINT_OUTPUT): print("PYTHON-NET: Input to trainOnBatch:")

        vectorMiniBatch = input_dict["vectorMiniBatch"]

        begTimeStep = np.array(vectorMiniBatch[0].begTimeStep)
        sampledTimeStep = np.array(vectorMiniBatch[0].sampledTimeStep)
        idx = sampledTimeStep - begTimeStep

        input_ = np.array(vectorMiniBatch[0].S)
        mu = np.array(vectorMiniBatch[0].MU)

        E, T, D = np.shape(input_)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)

        input_red = self.shiftIndexes(input_, idx, E)
        mu_red = self.shiftIndexes(mu, idx, E)

        input_red=np.array(input_red)
        mu_red=np.array(mu_red)

        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
        if not (T==self.BBTT_steps+1): raise ValueError("Number of time-steps does not match with BBTT_steps+1")

        output, sigma_ = self.forwardVector(input_red)
        if(PRINT_OUTPUT): print("PYTHON-NET: output = ", output)
        output_det = output.detach().numpy()
        action_sigma = sigma_.detach().numpy()

        if(PRINT_OUTPUT): print("Computing action_dim")
        action_dim = (self.output_dim-1)

        if(PRINT_OUTPUT): print("Computing action")
        action_mean = output_det[:,1:1+action_dim]
        action = np.random.normal(loc=action_mean,scale=action_sigma)

        value = output_det[:,0]
        # print("VALUES:")
        # print(np.shape(action))
        # print(np.shape(value))
        # print(np.shape(action_sigma))

        # After truncating data to the current idx (training on time-step idx), then selecting the last mu (mu of the sampled time-step)
        mu = mu_red[:, -1, :]
        idx=np.shape(mu)[1]
        if not (idx%2==0): raise ValueError("Problem with the dimensionality of the behavior policy mu.")
        mu_mean = mu[:,:idx//2]
        mu_sigma = mu[:,idx//2:]
        # print(np.shape(mu_mean))
        # print(np.shape(mu_sigma))
        # print("#########")

        CmaxRet = input_dict["CmaxRet"]
        CinvRet = input_dict["CinvRet"]

        # print("########## PRINTING MU ##############")
        # print(mu)

        if np.any(mu_sigma<=0) or np.any(action_sigma<=0):
            raise ValueError("## A variance is zero during batch training, something is broken.")
        else:
            impWeight = self.evaluateImportanceWeight(action, action_mean, action_sigma, mu_mean, mu_sigma)
        isOff = self.isFarPolicy(impWeight, CmaxRet, CinvRet)


#     Rvec grad;
#     if(isOff) grad = offPolCorrUpdate(S, t, O, P, thrID);
#     else grad = compute(S, t, O, P, thrID);

#     if(thrID==0)  profiler->stop_start("BCK");
#     NET.setGradient(grad, bID, t); // place gradient onto output layer


# template<typename Advantage_t, typename Policy_t, typename Action_t>
# Rvec RACER<Advantage_t, Policy_t, Action_t>::
# compute(Episode& S, const Uint samp, const Rvec& outVec,
#         const Policy_t& POL, const Uint thrID) const

#   const auto ADV = prepare_advantage<Advantage_t>(outVec, &POL);
#   const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = outVec[VsID];

#   shift retrace-advantage with current V(s) estimate:

#   const Real A_RET = S.Q_RET[samp] - V_cur; ########################
#   const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
#   const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
#   # all these min(CmaxRet,rho_cur) have no effect with ReFer enabled
#   const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
#   const Rvec polG = policyGradient(S.policies[samp], POL, ADV, A_RET, thrID);
#   const Rvec penalG  = POL.div_kl_grad(S.policies[samp], -1);
#   #if(!thrID) cout<<dkl<<" s "<<print(S.states[samp])
#   #  <<" pol "<<print(POL.getVector())<<" mu "<<MU)
#   #  <<" act: "<<print(S.actions[samp])<<" pg: "<<print(polG)
#   #  <<" pen: "<<print(penalG)<<" fin: "<<print(finalG)<<endl;
#   #prepare Q with off policy corrections for next step:
#   const Real dAdv = updateRetrace(S, samp, A_cur, V_cur, rho);
#   # compute the gradient:
#   Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
#   gradient[VsID] = beta * Ver;
#   POL.finalize_grad(Utilities::weightSum2Grads(polG, penalG, beta), gradient);
#   ADV.grad(POL.sampAct, beta * Aer, gradient);
#   S.setMseDklImpw(samp, Ver*Ver, dkl, rho, CmaxRet, CinvRet);
#   # logging for diagnostics:
#   trainInfo->log(V_cur+A_cur, A_RET-A_cur, polG,penalG, {dAdv,rho}, thrID);
#   return gradient;


# template<typename Advantage_t, typename Policy_t, typename Action_t>
# Rvec RACER<Advantage_t, Policy_t, Action_t>::
# offPolCorrUpdate(Episode& S, const Uint t, const Rvec output,
#   const Policy_t& pol, const Uint thrID) const
# {
#   const auto adv = prepare_advantage<Advantage_t>(output, &pol);
#   const Real A_cur = adv.computeAdvantage(pol.sampAct);
#   // shift retrace-advantage with current V(s) estimate:
#   const Real A_RET = S.Q_RET[t] - output[VsID];
#   const Real Ver = std::min((Real)1, pol.sampImpWeight) * (A_RET-A_cur);
#   updateRetrace(S, t, A_cur, output[VsID], pol.sampImpWeight);
#   S.setMseDklImpw(t, Ver*Ver,pol.sampKLdiv,pol.sampImpWeight, CmaxRet,CinvRet);
#   const Rvec pg = pol.div_kl_grad(S.policies[t], beta-1);
#   // only non zero gradient is policy penalization
#   Rvec gradient = Rvec(networks[0]->nOutputs(), 0);
#   pol.finalize_grad(pg, gradient);
#   return gradient;
# }




    def isFarPolicy(self, W, C, invC):
        isOff_ = [(W[i]>C) or (W[i]<invC) for i in range(len(W))]
        # If C<=1 assume we never filter far policy samples
        isOff = [(C>1.0)and isOff_[i] for i in range(len(isOff_))]
        return isOff

    def evaluateImportanceWeight(self, action, action_mean, action_sigma, mu_mean, mu_sigma):
        polLogProb = self.evalLogProbability(action, action_mean, action_sigma)
        behaviorLogProb = self.evalLogProbability(action, mu_mean, mu_sigma)
        impWeight = polLogProb - behaviorLogProb
        # print("SHAPE OF impWeight ")
        # print(np.shape(impWeight))
        impWeight = [7.0 if(impWeight_i>7.0) else (-7.0 if impWeight_i < -7.0 else impWeight_i) for impWeight_i in impWeight]
        impWeight = np.array(impWeight)
        return np.exp(impWeight)

    def evalLogProbability(self, var, mean, sigma):
        p=np.zeros((np.shape(mean)[0]))
        for i in range(np.shape(mean)[1]):
            precision = 1.0 / (sigma[:,i]**2)
            # print(precision)
            p -= precision * (var[:,i]-mean[:,i])**2
            p += np.log( precision / 2.0 / np.pi )
            # print(p)
        # print("SHAPE OF P ")
        # print(np.shape(p))
        # print(p)
        return 0.5 * p

    def shiftIndexes(self, input_, idx, num_episodes):
        # As the input_ contains multiple time-steps of some episode and we are training on a specific time-step sampled at time idx, we need to shift the input_ data
        input_red = []
        for i in range(num_episodes):
            # item has shape [T x D]
            item = input_[i]
            # Taking only the relevant time-steps
            input_red.append(item[idx[i]-self.BBTT_steps:idx[i]+1])
        return input_red

    def selectAction(self, input_dict):
        # PRINT_OUTPUT=True
        if(PRINT_OUTPUT): print("PYTHON-NET: selectAction")
        if(PRINT_OUTPUT): print("PYTHON-NET: Input to selectAction:")

        vectorMiniBatch = input_dict["vectorMiniBatch"]
        action = input_dict["action"]
        mu = input_dict["mu"]

        if(PRINT_OUTPUT): print("PYTHON-NET: vectorMiniBatch = ", vectorMiniBatch)
        if(PRINT_OUTPUT): print("PYTHON-NET: action = ", action)
        if(PRINT_OUTPUT): print("PYTHON-NET: mu = ", mu)

        begTimeStep = np.array(vectorMiniBatch[0].begTimeStep)
        sampledTimeStep = np.array(vectorMiniBatch[0].sampledTimeStep)
        idx = sampledTimeStep - begTimeStep

        input_ = np.array(vectorMiniBatch[0].S)
        E, T, D = np.shape(input_)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
        input_red = self.shiftIndexes(input_, idx, E)

        input_red=np.array(input_red)
        E, T, D = np.shape(input_red)
        if(PRINT_OUTPUT): print("|| Episodes x Time steps x Dimensionality = ", E, T, D)
        if not (T==self.BBTT_steps+1): raise ValueError("Number of time-steps does not match with BBTT_steps+1")

        output, sigma = self.forwardVector(input_red)
        # print("PYTHON-NET: output = ", output)
        output_det = output.detach().numpy()
        sigma_det = sigma.detach().numpy()

        # print("Computing action_dim")
        action_dim = (self.output_dim-1)
        if not (action_dim==np.shape(np.array(action))[0]): raise ValueError("Problem with the dimension of the computed action (mean).")

        value = output_det[0,0]
        action_mean = output_det[0,1:1+action_dim]
        action_sigma = sigma_det[0]

        # print("Computing action")
        action_python = np.random.normal(loc=action_mean,scale=action_sigma)

        # print("Copying action")
        for i in range(action_dim):
            action[i]=action_python[i] 

        # print("Copying behavior")
        for i in range(action_dim):
            mu[i]=action_mean[i] 
        for j in range(action_dim):
            mu[action_dim+j]=action_sigma[i]
        # print("########## PRINTING MU ##############")
        # print(mu)
        if np.any(action_sigma==0): raise ValueError("ZERO STD ACTION DETECTED!!")

        # print("PYTHON-NET: action = ", action_python)
        return 0

    def forwardVector(self, input_):
        # PRINT_OUTPUT=True
        input_tensor = torch.DoubleTensor(input_)
        # SHAPE: [K, T, D]
        K, T, D = input_tensor.size()
        if(PRINT_OUTPUT): print("PYTHON-NET: Input shape = ", input_tensor.size())

        if not (D==self.input_dim): raise ValueError("Problem with provided input dimension.")
        if self.net_type == "FFNN":
            # Forward propagation in FFNN
            if not (T==1): raise ValueError("Problem with the number of time-steps. In FFNN should be one (BBPT(0) + 1), but is {:}".format(T))
            # In FFNN the time-steps do not matter
            var = input_tensor.view(-1, self.input_dim)
            for nnLayer in self.fcLayers:
                var = F.relu(nnLayer(var))
            if(PRINT_OUTPUT): print("PYTHON-NET: Output shape = ", var.size())
        elif self.net_type == "LSTM":
            var = self.LSTM(input_tensor)
        else:
            raise ValueError("Invalid net-type")

        sp = nn.Softplus()
        sigma = sp(self.sigma_weights)
        sigma = sigma[None]
        sigma = torch.cat(K*[sigma])
        return var, sigma

if __name__ == "__main__":

    output_dim = 3
    sigma_dim = 2
    batch_size = 7
    BBTT_steps = 10
    input_dim = 5
    nnLayerSizes = [20, 20, 20]
    net_type = "LSTM"
    # net_type = "FFNN"

    input_ = torch.randn([batch_size, BBTT_steps+1, input_dim])

    net = NET(net_type, BBTT_steps, input_dim, nnLayerSizes, output_dim, sigma_dim)
    print(net)

    output, sigma = net.forwardVector(input_)

    input_ = torch.DoubleTensor(input_)
    print("Input size KxTxD")
    print(input_.size())
    print("Output size KxN")
    print(output.size())
    print("Sigma size KxN")
    print(sigma.size())

    # if(PRINT_OUTPUT): print(dir(output))
    # if(PRINT_OUTPUT): print(output.__dict__)
    # if(PRINT_OUTPUT): print(output)
    # if(PRINT_OUTPUT): print(output.detach().numpy())


