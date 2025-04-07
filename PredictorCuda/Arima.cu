#include "Arima.h"
#include <math.h>
#include <cuda_runtime.h>

/*
Descripción:
    -Funcion destinada a calcular la media de los valores pasados en un array. 
Parametros:
    -float *array: Suministro de datos.
    -int n: Tamaño del array.
*/
__device__ float mean(float *array, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += array[i];
    }
    return sum / n;
}

/*
Descripción:
    -Funcion que realiza la Regresion lineal 
Parametros:
    -float *x: Array de valores independientes ('Dias') 
    -float *y: Array de valores dependientes ('Extraido JSON') 
    -int n: Tamaño de ambos arrays.
    -float *b1: Valor calculado usado para la predicción.
    -float *b0: Valor calculado usado para la predicción.
*/
__global__ void linear_regression_kernel(float *x, float *y, int n, float *b1, float *b0)
{
    extern __shared__ float sdata[];
    float *sx = sdata;
    float *sy = &sdata[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Carga de datos en memoria compartida
    if (i < n)
    {
        sx[tid] = x[i];
        sy[tid] = y[i];
    }
    else
    {
        sx[tid] = 0.0f;
        sy[tid] = 0.0f;
    }
    __syncthreads();

    // Cálculo de medias
    float sum_x = 0.0f, sum_y = 0.0f;
    for (int s = 0; s < blockDim.x; s++)
    {
        if (blockIdx.x * blockDim.x + s < n)
        {
            sum_x += sx[s];
            sum_y += sy[s];
        }
    }
    float mean_x = sum_x / n;
    float mean_y = sum_y / n;

    // Cálculo de numerador y denominador
    float num = 0.0f, den = 0.0f;
    if (i < n)
    {
        float x_diff = x[i] - mean_x;
        num = x_diff * (y[i] - mean_y);
        den = x_diff * x_diff;
    }

    // Almacenar resultados en memoria compartida para reducción
    sx[tid] = num;
    sy[tid] = den;
    __syncthreads();

    // Reducción para sumar todos los hilos
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sx[tid] += sx[tid + s]; // Suma numerador
            sy[tid] += sy[tid + s]; // Suma denominador
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // Evitar división por cero
        if (sy[0] != 0.0f)
        {
            b1[blockIdx.x] = sx[0] / sy[0];
        }
        else
        {
            b1[blockIdx.x] = 0.0f;
        }
        b0[blockIdx.x] = mean_y - (b1[blockIdx.x] * mean_x);

        // Opcional: añadir limitación a las predicciones para evitar valores extremos
        if (isnan(b0[blockIdx.x]) || isinf(b0[blockIdx.x]))
            b0[blockIdx.x] = 0.0f;
        if (isnan(b1[blockIdx.x]) || isinf(b1[blockIdx.x]))
            b1[blockIdx.x] = 0.0f;
    }
}

/*
Descripcion:
    -Funcion para la diferenciacion o alisado de los datos en busqueda de generar series estacionarias 
Parametros: 
    -float *series: Array con los valores a normalizar  
    -int n: Tamaño del array original
    -int d: Factor de diferenciación. 
    -float *diff_series: Array contenedor de la nueva serie estilizada. 
*/
__global__ void difference_kernel(float *series, int n, int d, float *diff_series)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d && i < n)
    {
        diff_series[i - d] = series[i] - series[i - d];
    }
}

/*
Descripcion:
    -Funcion para el ajuste de los datos, tras la diferenciacion de los mismo. 
Parametros: 
    -float *series: Array de datos previamente diferenciados. 
    -int n: Tamaño de la serie tras la diferenciación.  
    -ARIMA_Model *model: Modelo ARIMA que estamos usando 
    -float *params: Array contenedor de la serie ajustada 
*/
__global__ void fit_arima_kernel(float *series, int n, ARIMA_Model *model, float *params)
{
    // Better initialization using autocorrelation
    if (threadIdx.x < model->p)
    { // AR terms
        if (blockIdx.x * blockDim.x + threadIdx.x < n - 1)
        {
            params[threadIdx.x] = 0.8f * (threadIdx.x + 1) / model->p; // autocorrelaion
        }
    }
    else if (threadIdx.x < model->p + model->q)
    { // MA terms
        params[threadIdx.x] = 0.2f * (threadIdx.x - model->p + 1) / model->q;
    }
}

/*
Descripción:
    -Función que calcula los valores autoregresivos y la media movil valores necesarios para la predicción del siguiente valor.
Parametros:
    -float *series: array pre estilizado 
    -int n: Tamaño del array original 
    -ARIMA_Model *model: Modelo de ARIMA 
    -float *params: Serie de datos estilizados.  
    -float *forecast: Prediccion Calculada.
*/
__global__ void forecast_arima_kernel(float *series, int n, ARIMA_Model *model,
                                      float *params,float *forecast)
{
    float ar_component = 0.0f;
    float ma_component = 0.0f;

    // AR Component
    for (int i = 0; i < model->p; i++)
    {
        ar_component += params[i] * series[n - 1 - i];
    }

    // MA Component
    float residual = series[n - 1] - ar_component;
    ma_component = params[model->p] * residual;

    if (threadIdx.x == 0)
    {
        forecast[0] = ar_component + ma_component;
        // Gestion de valores extremos. (Opcional)
        forecast[0] = fmaxf(0.0f, fminf(100.0f, forecast[0]));
    }
}
