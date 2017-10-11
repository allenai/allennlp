#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#include <stdio.h>
#include "highway_lstm_kernel.h"

#define BLOCK 256

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

__forceinline__ __device__ float dsigmoidf(float in) {
   float s = sigmoidf(in);
   return s * (1.f - s);
}

__forceinline__ __device__ float tanh2f(float in) {
   float t = tanhf(in);
   return t*t;
}

__global__ void elementWise_bp(int hiddenSize, int miniBatch, int numCovered,
                               // Inputs
                               float *out_grad,
                               float *h_out_grad,
                               float *c_out_grad,
                               float *c_in,
                               float *c_out,
                               float *h_out,
                               float *gates_out,
                               float *dropout_in,
                               // Outputs
                               float *c_in_grad,
                               float *i_gates_grad,
                               float *h_gates_grad,
                               int training) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (index >= numCovered * hiddenSize) return;
    
   int batch = index / hiddenSize;
   int h_gateIndex = (index % hiddenSize) + 5 * batch * hiddenSize;
   int i_gateIndex = (index % hiddenSize) + 6 * batch * hiddenSize;   

   float d_h = out_grad[index] + h_out_grad[index];
   d_h = d_h * dropout_in[index];

   float in_gate = gates_out[i_gateIndex];
   float forget_gate = gates_out[i_gateIndex + 1 * hiddenSize];
   float act_gate = gates_out[i_gateIndex + 2 * hiddenSize];
   float out_gate = gates_out[i_gateIndex + 3 * hiddenSize];
   float r_gate = gates_out[i_gateIndex + 4 * hiddenSize];
   float lin_gate = gates_out[i_gateIndex + 5 * hiddenSize];

   float d_out = d_h * r_gate;
   float d_c = d_out * out_gate * (1.f - tanh2f(c_out[index])) + c_out_grad[index];
   float h_prime = out_gate * tanhf(c_out[index]);

   float d_in_gate = d_c * act_gate * in_gate * (1.f - in_gate);
   float d_forget_gate = d_c * c_in[index] * forget_gate * (1.f - forget_gate);
   float d_act_gate = d_c * in_gate * (1.f - act_gate * act_gate);
   float d_out_gate = d_out * tanhf(c_out[index]) * out_gate * (1.f - out_gate);
   float d_r_gate = d_h * (h_prime - lin_gate) * r_gate * (1.f - r_gate);
   float d_lin_gate = d_h * (1 - r_gate);

   i_gates_grad[i_gateIndex] = d_in_gate;
   i_gates_grad[i_gateIndex + 1 * hiddenSize] = d_forget_gate;
   i_gates_grad[i_gateIndex + 2 * hiddenSize] = d_act_gate;
   i_gates_grad[i_gateIndex + 3 * hiddenSize] = d_out_gate;
   i_gates_grad[i_gateIndex + 4 * hiddenSize] = d_r_gate;
   i_gates_grad[i_gateIndex + 5 * hiddenSize] = d_lin_gate;

   h_gates_grad[h_gateIndex] = d_in_gate;
   h_gates_grad[h_gateIndex + 1 * hiddenSize] = d_forget_gate;
   h_gates_grad[h_gateIndex + 2 * hiddenSize] = d_act_gate;
   h_gates_grad[h_gateIndex + 3 * hiddenSize] = d_out_gate;
   h_gates_grad[h_gateIndex + 4 * hiddenSize] = d_r_gate;

   c_in_grad[index] = forget_gate * d_c;
}


// Fused forward kernel
__global__ void elementWise_fp(int hiddenSize, int miniBatch, int numCovered,
                               float *tmp_h, 
                               float *tmp_i, 
                               float *bias,
                               float *linearGates,
                               float *h_out,
                               float *dropout_in,
                               float *c_in,
                               float *c_out,
                               int training) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   if (index >= numCovered * hiddenSize) return;
   
   int batch = index / hiddenSize;
   int h_gateIndex = (index % hiddenSize) + 5 * batch * hiddenSize;
   int i_gateIndex = (index % hiddenSize) + 6 * batch * hiddenSize;   
   
   float g[6];

   for (int i = 0; i < 5; i++) {
      g[i] = tmp_i[i * hiddenSize + i_gateIndex] + tmp_h[i * hiddenSize + h_gateIndex];
      g[i] += bias[i * hiddenSize + index % hiddenSize];
   }   
   // extra for highway
   g[5] = tmp_i[5 * hiddenSize + i_gateIndex];
   
   float in_gate     = sigmoidf(g[0]);
   float forget_gate = sigmoidf(g[1]);
   float act_gate    = tanhf(g[2]);
   float out_gate    = sigmoidf(g[3]);
   float r_gate      = sigmoidf(g[4]);
   float lin_gate    = g[5];

   if (training == 1) {
       linearGates[i_gateIndex] = in_gate;
       linearGates[i_gateIndex + 1 * hiddenSize] = forget_gate;
       linearGates[i_gateIndex + 2 * hiddenSize] = act_gate;
       linearGates[i_gateIndex + 3 * hiddenSize] = out_gate;
       linearGates[i_gateIndex + 4 * hiddenSize] = r_gate;
       linearGates[i_gateIndex + 5 * hiddenSize] = lin_gate;
   }
   
   float val = (forget_gate * c_in[index]) + (in_gate * act_gate);
   
   c_out[index] = val;

   val = out_gate * tanhf(val);                                   
   val = val * r_gate + (1. - r_gate) * lin_gate;
   val = val * dropout_in[index];

   h_out[index] = val;
}

void highway_lstm_backward_ongpu(int inputSize, int hiddenSize, int miniBatch,
        int numLayers, int seqLength, float *out_grad, int *lengths,
        float *h_data_grad, float * c_data_grad, float *x, float *h_data,
        float *c_data, float *T,
        float *gates_out, float *dropout_in, float *h_gates_grad,
        float *i_gates_grad, float *h_out_grad, float *x_grad, float *T_grad, float *bias_grad,
        int isTraining, int do_weight_grad, cudaStream_t stream, cublasHandle_t handle) {


    const int numElements = hiddenSize * miniBatch;

    cudaStream_t stream_i;
    cudaStream_t stream_h;
    cudaStream_t stream_wi;
    cudaStream_t stream_wh;
    cudaStream_t stream_wb;

    cudaErrCheck(cudaStreamCreate(&stream_i));
    cudaErrCheck(cudaStreamCreate(&stream_h));
    cudaErrCheck(cudaStreamCreate(&stream_wi));
    cudaErrCheck(cudaStreamCreate(&stream_wh));
    cudaErrCheck(cudaStreamCreate(&stream_wb));

    float one = 1.f;
    float zero = 0.f;

    float *ones_host = new float[miniBatch];
    for (int i=0; i < miniBatch; i++) {
        ones_host[i] = 1.f;
    }
    float *ones;
    cudaErrCheck(cudaMalloc((void**)&ones, miniBatch * sizeof(float)));
    cudaErrCheck(cudaMemcpy(ones, ones_host, miniBatch * sizeof(float), cudaMemcpyHostToDevice));

    for (int layer = numLayers-1; layer >= 0; layer--) {
        int direction;
        int startInd;
        int currNumCovered;
        if (layer % 2 == 0) {
            // forward direction
            direction = -1;
            startInd = seqLength-1;
            currNumCovered = 0;
        } else {
            // backward direction
            direction = 1;
            startInd = 0;
            currNumCovered = miniBatch;
        }

        for (int t = startInd; t < seqLength && t >= 0; t = t + direction) {
            
            int prevIndex;
            int prevGradIndex;
            if (direction == 1) {
                while (lengths[currNumCovered-1] <= t) {
                    currNumCovered--;
                }
                prevGradIndex = t;
                prevIndex = (t+2)%(seqLength+1);
            } else {
                while ((currNumCovered < miniBatch) && (lengths[currNumCovered] > t)) {
                    currNumCovered++;
                }
                prevGradIndex = (t+2)%(seqLength+1);
                prevIndex = t;
            }


            float * gradPtr;
            if (layer == numLayers-1) {
                gradPtr = out_grad + t * numElements;
            } else {
                gradPtr = h_out_grad + t * numElements + layer * seqLength * numElements;
            }

            cublasErrCheck(cublasSetStream(handle, stream_i));

            dim3 blockDim;
            dim3 gridDim;

            blockDim.x = BLOCK;
            gridDim.x = ((currNumCovered * hiddenSize) + blockDim.x - 1) / blockDim.x;               

            elementWise_bp <<< gridDim, blockDim , 0, stream>>> 
                (hiddenSize, miniBatch, currNumCovered,
                 gradPtr,
                 h_data_grad + prevGradIndex * numElements + layer * (seqLength + 1) * numElements,
                 c_data_grad + prevGradIndex * numElements + layer * (seqLength + 1) * numElements,
                 c_data + prevIndex * numElements + layer * (seqLength + 1) * numElements,
                 c_data + (t+1) * numElements + layer * (seqLength + 1) * numElements,
                 h_data + (t+1) * numElements + layer * (seqLength + 1) * numElements,
                 gates_out + t * 6 * numElements + layer * seqLength * 6 * numElements,
                 dropout_in + layer * numElements,
                 c_data_grad + (t+1) * numElements + layer * (seqLength + 1) * numElements,
                 i_gates_grad,
                 h_gates_grad,
                 isTraining);
               cudaErrCheck(cudaGetLastError());
               // END

             cudaErrCheck(cudaDeviceSynchronize());

             float *out_grad_ptr;
             int weightStart;
             int inSize;
             if (layer == 0) {
                 inSize = inputSize;
                 out_grad_ptr = x_grad + t * inputSize * miniBatch;
                 weightStart = 0;
             } else {
                 inSize = hiddenSize;
                 out_grad_ptr = h_out_grad + t * numElements + (layer-1) * seqLength * numElements;
                weightStart = 6 * hiddenSize * inputSize + 5 * hiddenSize * hiddenSize + (layer - 1) * 11 * hiddenSize * hiddenSize;
             }

             cublasErrCheck(cublasSgemm(handle,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         inSize, currNumCovered, 6*hiddenSize,
                         &one,
                         &T[weightStart],
                         6 * hiddenSize,
                         i_gates_grad,
                         6 * hiddenSize,
                         &zero,
                         out_grad_ptr,
                         inSize));

             cublasErrCheck(cublasSetStream(handle, stream_h));

             cublasErrCheck(cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        hiddenSize, currNumCovered, 5*hiddenSize,
                        &one,
                        &T[weightStart + 6*hiddenSize*inSize],
                        5 * hiddenSize,
                        h_gates_grad,
                        5 * hiddenSize,
                        &zero,
                        h_data_grad + (t+1) * numElements + layer * (seqLength+1) * numElements,
                        hiddenSize));

             if (do_weight_grad == 1) {
                 float *inputPtr;
                 if (layer == 0) {
                     inputPtr = x + t * inputSize * miniBatch;
                 } else {
                     inputPtr = h_data + (t+1) * numElements + (layer - 1) * (seqLength+1) * numElements;
                 }

                 cublasErrCheck(cublasSetStream(handle, stream_wi));

                 // Update i_weights
                 cublasErrCheck(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             6 * hiddenSize, inSize, currNumCovered,
                             &one,
                             i_gates_grad,
                             6 * hiddenSize,
                             inputPtr,
                             inSize,
                             &one,
                             &T_grad[weightStart],
                             6 * hiddenSize));

                 cublasErrCheck(cublasSetStream(handle, stream_wh));

                 // Update h_weights
                 cublasErrCheck(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             5 * hiddenSize, hiddenSize, currNumCovered,
                             &one,
                             h_gates_grad,
                             5 * hiddenSize,
                             h_data + prevIndex * numElements + layer * (seqLength+1) * numElements,
                             hiddenSize,
                             &one,
                             &T_grad[weightStart + 6 *hiddenSize*inSize],
                             5 * hiddenSize));

                 cublasErrCheck(cublasSetStream(handle, stream_wb));

                 // Update bias_weights
                 cublasErrCheck(cublasSgemv(handle,
                             CUBLAS_OP_N,
                             5 * hiddenSize, currNumCovered,
                             &one,
                             h_gates_grad,
                             5 * hiddenSize,
                             ones,
                             1,
                             &one,
                             &bias_grad[layer * 5 * hiddenSize],
                             1));
             }

           cudaErrCheck(cudaDeviceSynchronize());

        }

    }

   cublasErrCheck(cublasSetStream(handle, stream));
   cudaErrCheck(cudaStreamDestroy(stream_i));
   cudaErrCheck(cudaStreamDestroy(stream_h));
   cudaErrCheck(cudaStreamDestroy(stream_wi));
   cudaErrCheck(cudaStreamDestroy(stream_wh));
   cudaErrCheck(cudaStreamDestroy(stream_wb));

   cudaErrCheck(cudaFree(ones));
   delete [] ones_host;

   cudaErrCheck(cudaDeviceSynchronize());
}

void highway_lstm_forward_ongpu(int inputSize, int hiddenSize, int miniBatch, 
        int numLayers, int seqLength, float *x, int *lengths, float *h_data, 
        float *c_data, float *tmp_i, float *tmp_h, float *T, float *bias,
        float *dropout, float *gates, int is_training, cudaStream_t stream, cublasHandle_t handle) {

    const int numElements = hiddenSize * miniBatch;

    float zero = 0.f;
    float one = 1.f;

    cudaStream_t stream_i;
    cudaStream_t stream_h;

    cudaErrCheck(cudaStreamCreate(&stream_i));
    cudaErrCheck(cudaStreamCreate(&stream_h));

    for (int layer = 0; layer < numLayers; layer++) {
        int direction;
        int startInd;
        int currNumCovered;
        if (layer % 2 == 0) {
            // forward direction
            direction = 1;
            startInd = 0;
            currNumCovered = miniBatch;
        } else {
            // backward direction
            direction = -1;
            startInd = seqLength-1;
            currNumCovered = 0;
        }
        cublasErrCheck(cublasSetStream(handle, stream));

        for (int t = startInd; t < seqLength && t >= 0; t = t + direction) {
            
            int prevIndex;
            if (direction == 1) {
                while (lengths[currNumCovered-1] <= t) {
                    currNumCovered--;
                }
                prevIndex = t;
            } else {
                while ((currNumCovered < miniBatch) && (lengths[currNumCovered] > t)) {
                    currNumCovered++;
                }
                prevIndex = (t+2)%(seqLength+1);
            }

            int inSize;
            int weightStart;
            float *inputPtr;
            if (layer == 0) {
                inSize = inputSize;
                weightStart = 0;
                inputPtr = x + t * inputSize * miniBatch;
                prevIndex = t;
            } else {
                inSize = hiddenSize;
                weightStart = 6 * hiddenSize * inputSize + 5 * hiddenSize * hiddenSize + (layer - 1) * 11 * hiddenSize * hiddenSize;
                inputPtr = h_data + (t+1) * numElements + (layer - 1) * (seqLength+1) * numElements;
            }

            cublasErrCheck(cublasSetStream(handle, stream_i));

            cublasErrCheck(cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        6*hiddenSize, currNumCovered, inSize,
                        &one,
                        &T[weightStart],
                        6 * hiddenSize,
                        inputPtr,
                        inSize,
                        &zero,
                        tmp_i,
                        6 * hiddenSize));

            cublasErrCheck(cublasSetStream(handle, stream_h));

            cublasErrCheck(cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        5*hiddenSize, currNumCovered, hiddenSize,
                        &one,
                        &T[6 * hiddenSize * inSize + weightStart],
                        5 * hiddenSize,
                        h_data + prevIndex * numElements + layer * (seqLength + 1) * numElements,
                        hiddenSize,
                        &zero,
                        tmp_h,
                        5 * hiddenSize));

            cudaErrCheck(cudaDeviceSynchronize());

            dim3 blockDim;
            dim3 gridDim;

            blockDim.x = BLOCK;
            gridDim.x = ((currNumCovered * hiddenSize) + blockDim.x - 1) / blockDim.x;               
            elementWise_fp <<< gridDim, blockDim , 0, stream>>> 
                (hiddenSize, miniBatch, currNumCovered,
                 tmp_h, 
                 tmp_i, 
                 bias + 5 * layer * hiddenSize,
                 is_training ? gates + 6 * (t * numElements + layer * seqLength * numElements) : NULL,
                 h_data + (t + 1) * numElements + layer * (seqLength + 1) * numElements,
                 dropout + layer * numElements,
                 c_data + prevIndex * numElements + layer * (seqLength + 1) * numElements,
                 c_data + (t + 1) * numElements + layer * (seqLength + 1) * numElements,
                 is_training);
               cudaErrCheck(cudaGetLastError());

            cudaErrCheck(cudaDeviceSynchronize());
        }
    }

   cublasErrCheck(cublasSetStream(handle, stream));
   cudaErrCheck(cudaStreamDestroy(stream_i));
   cudaErrCheck(cudaStreamDestroy(stream_h));

   cudaErrCheck(cudaDeviceSynchronize());
}

#ifdef __cplusplus
}
#endif
