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

void cambiar_comas_por_puntos(char *str) {
    for (int i = 0; str[i]; i++) {
        if (str[i] == ',') str[i] = '.';
    }
}

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
    char *file_data = NULL;
    long file_length = 0;
    int num_campos = argc - 3;
    char (*campos)[32] = NULL;
    cJSON *root = NULL;
    int total_registros = 0;
    float **valores_completos = NULL;
    CampoDatosHibrido *datos_locales = NULL;
    int nh = atoi(argv[2]);

    // Paso 1: Solo el proceso 0 lee y parsea el archivo
    if (rank == 0) {
        FILE *file = fopen(argv[1], "rb");
        if (!file) {
            perror("Error al abrir el archivo");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        fseek(file, 0, SEEK_END);
        file_length = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        file_data = (char *)malloc(file_length + 1);
        if (!file_data) {
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        if (fread(file_data, 1, file_length, file) != file_length) {
            free(file_data);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fclose(file);
        file_data[file_length] = '\0';

        // Parsear JSON solo en rank 0
        root = cJSON_Parse(file_data);
        if (!root) {
            fprintf(stderr, "Error al parsear JSON\n");
            free(file_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        total_registros = cJSON_GetArraySize(root);
    }

    // Broadcast número total de registros
    MPI_Bcast(&total_registros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_registros == 0) {
        if (rank == 0) {
            fprintf(stderr, "No hay registros para procesar\n");
        }
        if (rank == 0) {
            cJSON_Delete(root);
            free(file_data);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Broadcast número de campos
    MPI_Bcast(&num_campos, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Proceso 0 prepara nombres de campos
    if (rank == 0) {
        campos = malloc(num_campos * 32 * sizeof(char));
        if (!campos) {
            cJSON_Delete(root);
            free(file_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        for (int i = 0; i < num_campos; i++) {
            strncpy(campos[i], argv[i + 3], 31);
            campos[i][31] = '\0';
        }
    } else {
        campos = malloc(num_campos * 32 * sizeof(char));
        if (!campos) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast nombres de campos
    MPI_Bcast(campos, num_campos * 32, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Distribución de trabajo optimizada
    int registros_por_rank = total_registros / size;
    int resto = total_registros % size;
    int inicio = rank * registros_por_rank + (rank < resto ? rank : resto);
    int fin = inicio + registros_por_rank + (rank < resto ? 1 : 0);
    int registros_locales = fin - inicio;

    // Configurar OpenMP
    omp_set_num_threads(nh);

    // Preparar estructuras para datos locales
    datos_locales = malloc(num_campos * sizeof(CampoDatosHibrido));
    if (!datos_locales) {
        if (rank == 0) {
            cJSON_Delete(root);
            free(file_data);
        }
        free(campos);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < num_campos; i++) {
        strncpy(datos_locales[i].nombre, campos[i], 31);
        datos_locales[i].nombre[31] = '\0';
        datos_locales[i].valores = malloc(registros_locales * sizeof(float));
        datos_locales[i].cantidad = registros_locales;
        
        if (!datos_locales[i].valores) {
            for (int j = 0; j < i; j++) free(datos_locales[j].valores);
            free(datos_locales);
            if (rank == 0) {
                cJSON_Delete(root);
                free(file_data);
            }
            free(campos);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Proceso 0 envía los datos necesarios a cada proceso
    if (rank == 0) {
        // Procesar su propia parte primero
        #pragma omp parallel for
        for (int i = 0; i < num_campos; i++) {
            for (int j = 0; j < registros_locales; j++) {
                cJSON *item = cJSON_GetArrayItem(root, inicio + j);
                if (item) {
                    cJSON *valor = cJSON_GetObjectItem(item, campos[i]);
                    datos_locales[i].valores[j] = convertir_valor(valor);
                } else {
                    datos_locales[i].valores[j] = 0.0f;
                }
            }
        }

        // Enviar datos a otros procesos
        for (int p = 1; p < size; p++) {
            int p_inicio = p * registros_por_rank + (p < resto ? p : resto);
            int p_fin = p_inicio + registros_por_rank + (p < resto ? 1 : 0);
            int p_registros = p_fin - p_inicio;

            for (int i = 0; i < num_campos; i++) {
                float *temp_buffer = malloc(p_registros * sizeof(float));
                
                #pragma omp parallel for
                for (int j = 0; j < p_registros; j++) {
                    cJSON *item = cJSON_GetArrayItem(root, p_inicio + j);
                    if (item) {
                        cJSON *valor = cJSON_GetObjectItem(item, campos[i]);
                        temp_buffer[j] = convertir_valor(valor);
                    } else {
                        temp_buffer[j] = 0.0f;
                    }
                }

                MPI_Send(temp_buffer, p_registros, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                free(temp_buffer);
            }
        }
    } else {
        // Otros procesos reciben sus datos
        for (int i = 0; i < num_campos; i++) {
            MPI_Recv(datos_locales[i].valores, registros_locales, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Liberar memoria del JSON en rank 0
    if (rank == 0) {
        cJSON_Delete(root);
        free(file_data);
    }

    // Proceso 0 prepara buffers para reunir los datos
    if (rank == 0) {
        valores_completos = malloc(num_campos * sizeof(float *));
        if (!valores_completos) {
            free(campos);
            for (int i = 0; i < num_campos; i++) free(datos_locales[i].valores);
            free(datos_locales);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < num_campos; i++) {
            valores_completos[i] = malloc(total_registros * sizeof(float));
            if (!valores_completos[i]) {
                for (int j = 0; j < i; j++) free(valores_completos[j]);
                free(valores_completos);
                free(campos);
                for (int j = 0; j < num_campos; j++) free(datos_locales[j].valores);
                free(datos_locales);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    // Reunir los datos en el proceso 0
    int *recv_counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        int i_start = i * registros_por_rank + (i < resto ? i : resto);
        int i_end = i_start + registros_por_rank + (i < resto ? 1 : 0);
        recv_counts[i] = i_end - i_start;
        displs[i] = i_start;
    }

    for (int i = 0; i < num_campos; i++) {
        MPI_Gatherv(
            datos_locales[i].valores,
            registros_locales,
            MPI_FLOAT,
            rank == 0 ? valores_completos[i] : NULL,
            recv_counts,
            displs,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
        );
    }

    // Procesamiento en el proceso 0
    if (rank == 0) {
        float *dias_transcurridos = array_auxiliar_dias(total_registros);
        
        // Regresión Lineal para cada campo
        for (int i = 0; i < num_campos; i++) {
            float b0, b1;
            linear_regression(dias_transcurridos, valores_completos[i], total_registros, &b1, &b0);
            float prediccion = b0 + b1 * (total_registros + 1);
            printf("Regresión Lineal - Previsión %s día %d: %.2f\n", 
                   campos[i], total_registros + 1, prediccion);
        }
        
        // Modelos ARIMA
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
                
                fit_arima(valores_completos[i], total_registros, &modelo, params);
                float prediccion = forecast_arima(valores_completos[i], total_registros, &modelo, params);
                printf("  Campo %s - Previsión día %d: %.2f\n", 
                       campos[i], total_registros + 1, prediccion);
                
                free(params);
            }
        }

        free(dias_transcurridos);
        for (int i = 0; i < num_campos; i++) {
            free(valores_completos[i]);
        }
        free(valores_completos);
    }

    // Limpieza
    for (int i = 0; i < num_campos; i++) {
        free(datos_locales[i].valores);
    }
    free(datos_locales);
    free(campos);
    free(recv_counts);
    free(displs);

    MPI_Finalize();
    return EXIT_SUCCESS;
}