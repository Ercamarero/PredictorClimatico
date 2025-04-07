#ifndef ARIMA_H
#define ARIMA_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int p;  // Orden AR (Autoregresivo) // Cantidad de valores previos que atendemos 
    int d;  // Orden de diferenciación  // Como manejamos la tendencia 
    int q;  // Orden MA (Media Móvil) // analisis de fluctuaciones recientes 
} ARIMA_Model;

// Funciones básicas
float mean(float *array, int n);
void linear_regression(float *x, float *y, int n, float *b1, float *b0);

// Funciones ARIMA
void difference(float *series, int n, int d, float *diff_series);
void fit_arima(float *series, int n, ARIMA_Model *model, float *params);
float forecast_arima(float *series, int n, ARIMA_Model *model, float *params);

#endif