#include <cublasXt.h>

#ifdef __cplusplus
extern "C" {
#endif

void highway_lstm_forward_ongpu(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength, float *x, int *lengths, float*h_data, float *c_data, float *tmp_i, float *tmp_h, float *T, float *bias, float *dropout, float *gates, int is_training, cudaStream_t stream, cublasHandle_t handle);

void highway_lstm_backward_ongpu(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength, float *out_grad, int *lengths, float *h_data_grad, float *c_data_grad, float *x, float *h_data, float *c_data, float *T, float *gates_out, float *dropout_in, float *h_gates_grad, float *i_gates_grad, float *h_out_grad, float *x_grad, float *T_grad, float *bias_grad, int isTraining, int do_weight_grad, cudaStream_t stream, cublasHandle_t handle);

#ifdef __cplusplus
}
#endif
