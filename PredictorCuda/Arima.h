#ifndef ARIMACUDA_H
#define ARIMACUDA_H

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int p; // Orden AR (Autoregresivo)
    int d; // Orden de diferenciación
    int q; // Orden MA (Media Móvil)
} ARIMA_Model;

// Funciones básicas
__global__ void linear_regression_kernel(float *x, float *y, int n, float *b1, float *b0);

// Funciones ARIMA
__global__ void difference_kernel(float *series, int n, int d, float *diff_series);
__global__ void fit_arima_kernel(float *series, int n, ARIMA_Model *model, float *params);
__global__ void forecast_arima_kernel(float *series, int n, ARIMA_Model *model, float *params, float *forecast);
#endif