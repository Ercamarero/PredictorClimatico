#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include "Arima.h"

typedef struct
{
    char *nombre;
    float *valores;
    int cantidad;
} CampoDatos;

/*
Descripción:
    -Funcion para estilizar los datos  del JSON para que se puedan usar.
Parametros:
    -char *str: cadena
*/
void cambiar_comas_por_puntos(char *str)
{
    for (int i = 0; str[i]; i++)
    {
        if (str[i] == ',')
            str[i] = '.';
    }
}

/*
Descripción:
    -Funcion que genera la variable independiente en este caso los dias
Parametros:
    -int total: Cantidad de registros en nuestro JSON
*/
float *array_auxiliar_dias(int total)
{
    float *dias = (float *)malloc(total * sizeof(float));
    if (dias == NULL)
    {
        fprintf(stderr, "Error al asignar memoria para días\n");
        return NULL;
    }

#pragma omp parallel for
    for (int i = 0; i < total; i++)
    {
        dias[i] = (float)(i + 1);
    }
    return dias;
}
/*
Descripción:
    -Funcion para la extraccion de los valores del JSON de datos que le asignamos.
Parametros:
    -cJSON *array_json: Fichero JSON preParseado para la actual extraccion de valores.
    -char **campos: Valores que estamos buscando dentro del JSON parseado.
    -int num_campos: Cantidad de campos de datos que queremos obtener.
    -int *total_registros Tamaño de segmentos que tiene nuestro JSON.
*/
CampoDatos *extraer_campos_openmp(cJSON *array_json, char **campos, int num_campos, int *total_registros)
{
    *total_registros = cJSON_GetArraySize(array_json);
    CampoDatos *resultados = (CampoDatos *)malloc(num_campos * sizeof(CampoDatos));

#pragma omp parallel for
    for (int i = 0; i < num_campos; i++)
    {
        resultados[i].nombre = strdup(campos[i]);
        resultados[i].valores = (float *)malloc(*total_registros * sizeof(float));
        resultados[i].cantidad = *total_registros;

#pragma omp parallel for
        for (int j = 0; j < *total_registros; j++)
        {
            // Obtiene el conjunto completo contenido en {}
            cJSON *item = cJSON_GetArrayItem(array_json, j);
            // obtiene la correlacion clave valor con respecto al campo buscado
            cJSON *valor = cJSON_GetObjectItem(item, campos[i]);

            if (valor)
            {
                if (cJSON_IsNumber(valor))
                {
                    resultados[i].valores[j] = (float)valor->valuedouble;
                }
                else if (cJSON_IsString(valor))
                {
                    char str_valor[32];
                    strcpy(str_valor, valor->valuestring);
                    cambiar_comas_por_puntos(str_valor);
                    resultados[i].valores[j] = atof(str_valor);
                }
                else
                {
                    resultados[i].valores[j] = 0.0f;
                }
            }
            else
            {
                resultados[i].valores[j] = 0.0f;
            }
        }
    }

    return resultados;
}
/*
Descripción:
    - Funcion para el lanzamiento de  los kernels CUDA, ocupada de la reserva y asignacion de los recursos para que la GPU pueda ocuparse del trabajo.
Parametros:
    -float *d_dias: Array de enteros auxiliar para el cálculo de la regresión lineal, componente independiente del calculo.
    -float *d_registros: Valores tomados del JSON.
    -int n: Numero total de registros.
    -float *h_b0: Espacio en el host donde copiar el valor calculado en el device
    -float *h_b1: Espacio en el host donde copiar el valor calculado en el device

*/
void perform_linear_regression(float *d_dias, float *d_registros, int n, float *h_b0, float *h_b1)
{
    float *d_b0, *d_b1;

    // Allocate device memory for results
    checkCudaErrors(cudaMalloc((void **)&d_b0, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_b1, sizeof(float)));

    // Configure kernel launch parameters
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(float);

    // Launch kernel
    linear_regression_kernel<<<blocks, threads, shared_mem>>>(d_dias, d_registros, n, d_b1, d_b0);

    // Copy results back to host
    checkCudaErrors(cudaMemcpy(h_b0, d_b0, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b1, d_b1, sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_b0);
    cudaFree(d_b1);
}
/*
Descripción:
    - Funcion para el lanzamiento de  los kernels CUDA, ocupada de la reserva y asignacion de los recursos para que la GPU pueda ocuparse del trabajo.
Parametros:
    -float *d_series: Array de enteros que será la serie estilizada por la funcion difference
    -int n: Numero total de registros.
    -ARIMA_Model *model: Modelo de ARIMA que usaremos para la prediccion.
    -float *h_forecast : Espacio en el host donde copiar el valor calculado en el device
*/
void perform_arima_forecast(float *d_series, int n, ARIMA_Model *model, float *h_forecast)
{
    float *d_forecast, *d_params;
    ARIMA_Model *d_model;

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&d_forecast, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_params, (model->p + model->q) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_model, sizeof(ARIMA_Model)));

    // Copy model to device
    checkCudaErrors(cudaMemcpy(d_model, model, sizeof(ARIMA_Model), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threads = 256;
    int blocks = 1;
    size_t shared_mem = n * sizeof(float);

    // Fit ARIMA model
    fit_arima_kernel<<<(model->p + model->q + threads - 1) / threads, threads>>>(d_series, n, d_model, d_params);

    // Forecast
    forecast_arima_kernel<<<blocks, threads, shared_mem>>>(d_series, n, d_model, d_params, d_forecast);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_forecast, d_forecast, sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_forecast);
    cudaFree(d_params);
    cudaFree(d_model);
}

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    if (argc < 3)
    {
        printf("Uso: %s <archivo.json> <n_hilos> <campo1> <campo2> ...\n", argv[0]);
        return 1;
    }

    // Leer archivo JSON
    FILE *file = fopen(argv[1], "rb");
    if (!file)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = (char *)malloc(length + 1);
    fread(data, 1, length, file);
    fclose(file);
    data[length] = '\0';

    // Parsear JSON
    cJSON *root = cJSON_Parse(data);
    if (!root)
    {
        fprintf(stderr, "Error al parsear JSON\n");
        free(data);
        return 1;
    }

    // Configurar OpenMP
    int n_hilos = atoi(argv[2]);
    omp_set_num_threads(n_hilos);

    // Extraer datos en paralelo
    int num_campos = argc - 3;
    char **campos = argv + 3;
    int total_registros;

    // Llamamiento a las funciones principales.
    CampoDatos *datos = extraer_campos_openmp(root, campos, num_campos, &total_registros);
    float *h_dias = array_auxiliar_dias(total_registros);

    // Reservar memoria en el device
    float *d_dias;
    checkCudaErrors(cudaMalloc((void **)&d_dias, total_registros * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_dias, h_dias, total_registros * sizeof(float), cudaMemcpyHostToDevice));

    // Procesar cada campo
    // Regresion Lineal
    for (int i = 0; i < num_campos; i++)
    {
        float *d_registros;
        checkCudaErrors(cudaMalloc((void **)&d_registros, total_registros * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_registros, datos[i].valores, total_registros * sizeof(float), cudaMemcpyHostToDevice));

        // Regresion Lineal
        float h_b0, h_b1;
        perform_linear_regression(d_dias, d_registros, total_registros, &h_b0, &h_b1);

        float prediccion = h_b0 + h_b1 * (total_registros + 1);
        printf("Regresión Lineal - Previsión %s día %d: %.2f\n",
               campos[i], total_registros + 1, prediccion);

    }

    // modelo ARIMA
    ARIMA_Model modelos_arima[] = {
        {1, 0, 0},
    };

    int num_modelos = sizeof(modelos_arima) / sizeof(modelos_arima[0]);

    for(int m = 0; m < num_modelos; m++)
    {   
        printf("Modelo ARIMA(%d,%d,%d)\n", modelos_arima[m].p, modelos_arima[m].d, modelos_arima[m].q);
        for(int i = 0; i < num_campos; i++)
        {
            float *d_registros;
            checkCudaErrors(cudaMalloc((void **)&d_registros, total_registros * sizeof(float)));
            checkCudaErrors(cudaMemcpy(d_registros, datos[i].valores, total_registros * sizeof(float), cudaMemcpyHostToDevice));
            float arima_forecast;
            ARIMA_Model modelo = modelos_arima[m];
            perform_arima_forecast(d_registros, total_registros, &modelo, &arima_forecast);
            printf("Previsión %s día %d: %.2f\n",
                 campos[i], total_registros + 1, arima_forecast);
        }
    }
    // Liberamos memoria del device
    cudaFree(d_dias);

    // Liberar memoria
    for (int i = 0; i < num_campos; i++)
    {
        free(datos[i].nombre);
        free(datos[i].valores);
    }
    free(datos);
    free(h_dias);
    cJSON_Delete(root);
    free(data);

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return 0;
}