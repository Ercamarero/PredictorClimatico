#include "Arima.h"
#include <math.h>
#include <omp.h>

/*
Descripción:
    -Funcion destinada a calcular la media de los valores pasados en un array. 
Parametros:
    -float *array: Suministro de datos.
    -int n: Tamaño del array.
*/
float mean(float *array, int n) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static, 8)
    for (int i = 0; i < n; i++) {
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
void linear_regression(float *x, float *y, int n, float *b1, float *b0) {
    float mean_x = mean(x, n);
    float mean_y = mean(y, n);
    float numerator = 0.0f, denominator = 0.0f;

    #pragma omp parallel for reduction(+:numerator, denominator) schedule(static, 8)
    for (int i = 0; i < n; i++) {
        float x_diff = x[i] - mean_x;
        numerator += x_diff * (y[i] - mean_y);
        denominator += x_diff * x_diff;
    }

    *b1 = numerator / denominator;
    *b0 = mean_y - (*b1 * mean_x);
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
void difference(float *series, int n, int d, float *diff_series) {
    #pragma omp parallel for schedule(static, 8)
    for (int i = d; i < n; i++) {
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
void fit_arima(float *series, int n, ARIMA_Model *model, float *params) {
    float *diff_series = malloc((n - model->d) * sizeof(float));
    difference(series, n, model->d, diff_series);

    #pragma omp parallel for
    for (int i = 0; i < model->p + model->q; i++) {
        params[i] = 0.5f * (i + 1) / (model->p + model->q);  
    }

    free(diff_series);
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
float forecast_arima(float *series, int n, ARIMA_Model *model, float *params) {
    float forecast = 0.0f;
    // Componente AR
    #pragma omp parallel for reduction(+:forecast) schedule(static, 8)
    for (int i = 0; i < model->p; i++) {
        forecast += params[i] * series[n - 1 - i]; 
    }
    // Componente MA 
    forecast += params[model->p] * (series[n-1] - mean(series, n)); 
    return forecast;
}
