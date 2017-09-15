int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength,
    THCudaTensor *x, THIntTensor *lengths, THCudaTensor *h_data,
    THCudaTensor *c_data, THCudaTensor *tmp_i,
    THCudaTensor *tmp_h, THCudaTensor *T, THCudaTensor *bias,
    THCudaTensor *dropout, THCudaTensor *gates, int isTraining);

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch, 
        int numLayers, int seqLength, THCudaTensor *out_grad, THIntTensor *lengths,
        THCudaTensor *h_data_grad, THCudaTensor *c_data_grad, THCudaTensor *x, 
        THCudaTensor *h_data, THCudaTensor *c_data, THCudaTensor *T,
        THCudaTensor *gates_out, THCudaTensor *dropout_in,
        THCudaTensor *h_gates_grad, THCudaTensor *i_gates_grad,
        THCudaTensor *h_out_grad, THCudaTensor *x_grad,  THCudaTensor *T_grad,
        THCudaTensor *bias_grad, int isTraining, int do_weight_grad);
