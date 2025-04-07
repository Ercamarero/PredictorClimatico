#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "cJSON.h"
#include <mpi.h>
#include <omp.h>
#include "Arima.h"

typedef struct {
    char nombre[32];
    float *valores;
    int cantidad;
} CampoDatosHibrido;

/*
Descripción:
    -Funcion para estilizar los datos  del JSON para que se puedan usar.
Parametros:
    -char *str: cadena
*/
void cambiar_comas_por_puntos(char *str) {
    for (int i = 0; str[i]; i++) {
        if (str[i] == ',') str[i] = '.';
    }
}

/*
Descripción:
    -Funcion que genera la variable independiente en este caso los dias
Parametros:
    -int total: Cantidad de registros en nuestro JSON 
*/
float *array_auxiliar_dias(int total) {
    float *dias = malloc(total * sizeof(float));
    if (dias == NULL) {
        fprintf(stderr, "Error al asignar memoria para días\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        dias[i] = (float)(i + 1);
    }
    return dias;
}

/*
Descripción:
    -Funcion para cambiar valores no numericos a formato float 
Parametros:
    -CJSON *valor: Fichero JSON a procesar 
*/
float convertir_valor(cJSON *valor) {
    if (valor == NULL) return 0.0f;

    if (cJSON_IsNumber(valor)) {
        return (float)valor->valuedouble;
    } else if (cJSON_IsString(valor)) {
        char str_valor[32];
        strncpy(str_valor, valor->valuestring, 31);
        str_valor[31] = '\0';
        cambiar_comas_por_puntos(str_valor);

        char *endptr;
        float num = strtof(str_valor, &endptr);
        if (endptr == str_valor || *endptr != '\0') {
            return 0.0f;
        }
        return num;
    }
    return 0.0f;
}

int main(int argc, char *argv[]) {
    //INICIO MPI 
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 && rank == 0) {
        printf("Uso: mpirun -np 4 %s <archivo.json> <N-hilos> <campo1> <campo2> ...\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    // Variables
    cJSON *root = NULL;
    char *data = NULL;
    long length = 0;
    float **valores_completos = NULL;
    float **valores_por_campo = NULL;
    int *send_counts = NULL;
    int *displs = NULL;
    int nh = atoi(argv[2]);
    //Hilo 0 apertura y parsing del fichero. 
    if (rank == 0) {
        FILE *file = fopen(argv[1], "rb");
        if (!file) {
            perror("Error al abrir el archivo");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fseek(file, 0, SEEK_END);
        length = ftell(file);
        fseek(file, 0, SEEK_SET);
        data = (char *)calloc(length + 1, sizeof(char));
        if (!data) {
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (fread(data, 1, length, file) != length) {
            free(data);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fclose(file);
        data[length] = '\0';
        root = cJSON_Parse(data);
        if (!root) {
            fprintf(stderr, "Error al parsear JSON\n");
            free(data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    //Establecer numero de campos a extraer.
    int num_campos = argc - 3;
    //Envio del numero de campos a los demas procesos 
    MPI_Bcast(&num_campos, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //El proceso 0 obtiene el nombre de los campos a extraer los demas en caso de error abortan.
    char (*campos)[32] = malloc(num_campos * 32 * sizeof(char));
    if (!campos) {
        if (rank == 0) {
            cJSON_Delete(root);
            free(data);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        for (int i = 0; i < num_campos; i++) {
            strncpy(campos[i], argv[i + 3], 31);
            campos[i][31] = '\0';
        }
    }
    //Enviamos los campos a buscar
    MPI_Bcast(campos, num_campos * 32, MPI_CHAR, 0, MPI_COMM_WORLD);

    //Marcamos la cantidad de registros del JSON 
    int total_registros = 0;
    if (rank == 0) {
        total_registros = cJSON_GetArraySize(root);
    }
    MPI_Bcast(&total_registros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //Si vacio = fin 
    if (total_registros == 0) {
        if (rank == 0) {
            fprintf(stderr, "No hay registros para procesar\n");
            cJSON_Delete(root);
            free(data);
            free(campos);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    //Reparto de las lineas entre los distintos procesos.
    int registros_por_rank = total_registros / size;
    int resto = total_registros % size;
    int inicio = rank * registros_por_rank + (rank < resto ? rank : resto);
    int fin = inicio + registros_por_rank + (rank < resto ? 1 : 0);
    int registros_locales = fin - inicio;

    //Asignacion de hilos OMP 
    omp_set_num_threads(nh);

    //Preproceso de los envios 
    if (rank == 0) {
        send_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if (!send_counts || !displs) {
            free(send_counts);
            free(displs);
            cJSON_Delete(root);
            free(data);
            free(campos);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < size; i++) {
            int i_start = i * registros_por_rank + (i < resto ? i : resto);
            int i_end = i_start + registros_por_rank + (i < resto ? 1 : 0);
            send_counts[i] = i_end - i_start;
            displs[i] = i_start;
        }

        valores_por_campo = malloc(num_campos * sizeof(float *));
        if (!valores_por_campo) {
            free(send_counts);
            free(displs);
            cJSON_Delete(root);
            free(data);
            free(campos);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < num_campos; i++) {
            valores_por_campo[i] = calloc(total_registros, sizeof(float));
            if (!valores_por_campo[i]) {
                for (int j = 0; j < i; j++) free(valores_por_campo[j]);
                free(valores_por_campo);
                free(send_counts);
                free(displs);
                cJSON_Delete(root);
                free(data);
                free(campos);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
    //Creamos los espacios para los struct que tendra cada proceso
    CampoDatosHibrido *datos_locales = malloc(num_campos * sizeof(CampoDatosHibrido));
    if (!datos_locales) {
        if (rank == 0) {
            for (int i = 0; i < num_campos; i++) free(valores_por_campo[i]);
            free(valores_por_campo);
            free(send_counts);
            free(displs);
            cJSON_Delete(root);
            free(data);
            free(campos);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_campos; i++) {
        strncpy(datos_locales[i].nombre, campos[i], 31);
        datos_locales[i].nombre[31] = '\0';
        datos_locales[i].valores = calloc(registros_locales, sizeof(float));
        datos_locales[i].cantidad = registros_locales;

        if (!datos_locales[i].valores) {
            for (int j = 0; j < i; j++) free(datos_locales[j].valores);
            free(datos_locales);
            if (rank == 0) {
                for (int j = 0; j < num_campos; j++) free(valores_por_campo[j]);
                free(valores_por_campo);
                free(send_counts);
                free(displs);
                cJSON_Delete(root);
                free(data);
                free(campos);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (rank == 0) {
        valores_completos = malloc(num_campos * sizeof(float *));
        if (!valores_completos) {
            for (int i = 0; i < num_campos; i++) {
                free(datos_locales[i].valores);
                free(valores_por_campo[i]);
            }
            free(datos_locales);
            free(valores_por_campo);
            free(send_counts);
            free(displs);
            cJSON_Delete(root);
            free(data);
            free(campos);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < num_campos; i++) {
            valores_completos[i] = malloc(total_registros * sizeof(float));
            if (!valores_completos[i]) {
                for (int j = 0; j < i; j++) free(valores_completos[j]);
                free(valores_completos);
                for (int j = 0; j < num_campos; j++) {
                    free(datos_locales[j].valores);
                    free(valores_por_campo[j]);
                }
                free(datos_locales);
                free(valores_por_campo);
                free(send_counts);
                free(displs);
                cJSON_Delete(root);
                free(data);
                free(campos);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            #pragma omp parallel for
            for (int j = 0; j < total_registros; j++) {
                cJSON *item = cJSON_GetArrayItem(root, j);
                if (item) {
                    cJSON *valor = cJSON_GetObjectItem(item, campos[i]);
                    valores_completos[i][j] = convertir_valor(valor);
                } else {
                    valores_completos[i][j] = 0.0f;
                }
            }
        }
    }
    //Envio y recoleccion
    for (int i = 0; i < num_campos; i++) {
        MPI_Scatterv(
            rank == 0 ? valores_completos[i] : NULL,
            send_counts,
            displs,
            MPI_FLOAT,
            datos_locales[i].valores,
            registros_locales,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );

        MPI_Gatherv(
            datos_locales[i].valores,
            registros_locales,
            MPI_FLOAT,
            rank == 0 ? valores_por_campo[i] : NULL,
            send_counts,
            displs,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );
    }
    //Procesamiento
    if (rank == 0) {
        for (int i = 0; i < num_campos; i++) {
            free(valores_completos[i]);
        }
        free(valores_completos);

        float *dias_transcurridos = array_auxiliar_dias(total_registros);
        
        // Linear Regression for each field
        for (int i = 0; i < num_campos; i++) {
            float b0, b1;
            linear_regression(dias_transcurridos, valores_por_campo[i], total_registros, &b1, &b0);
            float prediccion = b0 + b1 * (total_registros + 1);
            printf("Regresión Lineal - Previsión %s día %d: %.2f\n", 
                   campos[i], total_registros + 1, prediccion);
        }
        
        // ARIMA Models - Apply each model to all fields
        ARIMA_Model modelos_arima[] = {
            {1, 0, 0},  // AR(1)
        };
        
        int num_modelos = sizeof(modelos_arima) / sizeof(modelos_arima[0]);
        
        for (int m = 0; m < num_modelos; m++) {
            ARIMA_Model modelo = modelos_arima[m];
            printf("\nAplicando modelo ARIMA(%d,%d,%d) a todos los campos:\n", 
                   modelo.p, modelo.d, modelo.q);
            
            for (int i = 0; i < num_campos; i++) {
                float *params = malloc((modelo.p + modelo.q) * sizeof(float));
                if (!params) {
                    fprintf(stderr, "Error al asignar memoria para parámetros ARIMA\n");
                    continue;
                }
                
                fit_arima(valores_por_campo[i], total_registros, &modelo, params);
                float prediccion = forecast_arima(valores_por_campo[i], total_registros, &modelo, params);
                printf("  Campo %s - Previsión día %d: %.2f\n", 
                       campos[i], total_registros + 1, prediccion);
                
                free(params);
            }
        }

        free(dias_transcurridos);
        for (int i = 0; i < num_campos; i++) {
            free(valores_por_campo[i]);
        }
        free(valores_por_campo);
    }

    for (int i = 0; i < num_campos; i++) {
        free(datos_locales[i].valores);
    }
    free(datos_locales);
    free(campos);

    if (rank == 0) {
        free(send_counts);
        free(displs);
        cJSON_Delete(root);
        free(data);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}