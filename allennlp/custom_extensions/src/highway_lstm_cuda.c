#include <THC/THC.h>
#include "highway_lstm_kernel.h"

extern THCState *state;

int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch,
        int numLayers, int seqLength,
        THCudaTensor *x,
        THIntTensor *lengths,
        THCudaTensor *h_data,
        THCudaTensor *c_data,
        THCudaTensor *tmp_i,
        THCudaTensor *tmp_h,
        THCudaTensor *T,
        THCudaTensor *bias,
        THCudaTensor *dropout,
        THCudaTensor *gates,
        int isTraining) {

    float * x_ptr = THCudaTensor_data(state, x);
    int * lengths_ptr = THIntTensor_data(lengths);
    float * h_data_ptr = THCudaTensor_data(state, h_data);
    float * c_data_ptr = THCudaTensor_data(state, c_data);
    float * tmp_i_ptr = THCudaTensor_data(state, tmp_i);
    float * tmp_h_ptr = THCudaTensor_data(state, tmp_h);
    float * T_ptr = THCudaTensor_data(state, T);
    float * bias_ptr = THCudaTensor_data(state, bias);
    float * dropout_ptr = THCudaTensor_data(state, dropout);
    float * gates_ptr;
    if (isTraining == 1) {
        gates_ptr = THCudaTensor_data(state, gates);
    } else {
        gates_ptr = NULL;
    }

    cudaStream_t stream = THCState_getCurrentStream(state);
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);

    highway_lstm_forward_ongpu(inputSize, hiddenSize, miniBatch, numLayers, 
            seqLength, x_ptr, lengths_ptr, h_data_ptr, c_data_ptr, tmp_i_ptr,
            tmp_h_ptr, T_ptr, bias_ptr, dropout_ptr, gates_ptr,
            isTraining, stream, handle);

    return 1;

}

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength,
        THCudaTensor *out_grad,
        THIntTensor *lengths,
        THCudaTensor *h_data_grad,
        THCudaTensor *c_data_grad,
        THCudaTensor *x,
        THCudaTensor *h_data,
        THCudaTensor *c_data,
        THCudaTensor *T,
        THCudaTensor *gates_out,
        THCudaTensor *dropout_in,
        THCudaTensor *h_gates_grad,
        THCudaTensor *i_gates_grad,
        THCudaTensor *h_out_grad,
        THCudaTensor *x_grad,
        THCudaTensor *T_grad,
        THCudaTensor *bias_grad,
        int isTraining,
        int do_weight_grad) {

    float * out_grad_ptr = THCudaTensor_data(state, out_grad);
    int * lengths_ptr = THIntTensor_data(lengths);
    float * h_data_grad_ptr = THCudaTensor_data(state, h_data_grad);
    float * c_data_grad_ptr = THCudaTensor_data(state, c_data_grad);
    float * x_ptr = THCudaTensor_data(state, x);
    float * h_data_ptr = THCudaTensor_data(state, h_data);
    float * c_data_ptr = THCudaTensor_data(state, c_data);
    float * T_ptr = THCudaTensor_data(state, T);
    float * gates_out_ptr = THCudaTensor_data(state, gates_out);
    float * dropout_in_ptr = THCudaTensor_data(state, dropout_in);
    float * h_gates_grad_ptr = THCudaTensor_data(state, h_gates_grad);
    float * i_gates_grad_ptr = THCudaTensor_data(state, i_gates_grad);
    float * h_out_grad_ptr = THCudaTensor_data(state, h_out_grad);
    float * x_grad_ptr = THCudaTensor_data(state, x_grad);
    float * T_grad_ptr = THCudaTensor_data(state, T_grad);
    float * bias_grad_ptr = THCudaTensor_data(state, bias_grad);

    cudaStream_t stream = THCState_getCurrentStream(state);
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);

    highway_lstm_backward_ongpu(inputSize, hiddenSize, miniBatch, numLayers,
            seqLength, out_grad_ptr, lengths_ptr, h_data_grad_ptr, c_data_grad_ptr,
            x_ptr, h_data_ptr, c_data_ptr, T_ptr, gates_out_ptr, dropout_in_ptr,
            h_gates_grad_ptr, i_gates_grad_ptr, h_out_grad_ptr,
            x_grad_ptr, T_grad_ptr, bias_grad_ptr, isTraining, do_weight_grad,
            stream, handle);

    return 1;

}
