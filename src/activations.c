#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            int idx = i * m.cols + j;
            double x = m.data[idx];
            if(a == LOGISTIC){
                // DONE
                m.data[idx] = 1.0 / (1.0 + exp(-x));
            } else if (a == RELU){
                // DONE
                m.data[idx] = x < 0 ? 0.0 : x;
            } else if (a == LRELU){
                // DONE
                m.data[idx] = x < 0 ? x * 0.1 : x;
            } else if (a == SOFTMAX){
                // DONE
                m.data[idx] = exp(x);
            }
            sum += m.data[idx];
        }
        if (a == SOFTMAX) {
            // DONE: have to normalize by sum if we are using SOFTMAX
            for (j = 0; j < m.cols; ++j) {
                m.data[i * m.cols + j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            int idx = i * m.cols + j;
            double x = m.data[idx];
            // DONE: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                d.data[idx] *= x * (1 - x);
            } else if (a == RELU){
                d.data[idx] *= x < 0 ? 0.0 : 1.0;
            } else if (a == LRELU){
                d.data[idx] *= x < 0 ? 0.1 : 1.0;
            } else if (a == SOFTMAX){
                // DONE, but WHY?
                m.data[idx] = 1.0;
            }
        }
    }
}
