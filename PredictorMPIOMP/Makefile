# MakeFile para esta version 
# Variables 

CC = mpicc
CFLAGS = -fopenmp -O3
LDFLAGS = -lm 
TARGET = PredictorHibrido

# Parametros
NP ?= 4
DATOS ?= ./datos.json
CAMPOS ?= tmax tmed tmin prec velmedia
NHILOS ?= 4

# SOURCES
SRCS = JSONParserHibrido.c Arima.c cJSON.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "----------------------------------------"
	@echo "Ejecutado en $(NP) procesos"
	@echo "Input: $(DATOS)"
	@echo "Campos a analizar: $(CAMPOS)"
	@echo "Numero de Hilos por proceso: $(NHILOS)"
	@echo "----------------------------------------"
	mpirun -np $(NP) ./$(TARGET) $(DATOS) $(NHILOS) $(CAMPOS)

clean:
	rm -f $(TARGET) *.o

help:
	@echo "Opciones Disponibles:"
	@echo "  all       - Build (default)"
	@echo "  run       - Ejecutar (default)"
	@echo "  clean     - Limpieza"
	@echo ""
	@echo "Parametros de Ejecución:"
	@echo "  NP       - Numero de procesos MPI (default: 4)"
	@echo "  DATOS    - Input (default: datos.json)"
	@echo "  CAMPOS   - Campos analizados (default: tmax tmed prec velmedia)"
	@echo "  NHILOS   - Numero de Hilos por proceso (default: 4)"
	@echo "  Cambiable usando export VARIABLE=NuevoValor"