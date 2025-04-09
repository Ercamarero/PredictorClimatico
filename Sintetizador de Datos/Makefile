#Variables
CC = python3

#Parametros
DATOS ?= ./datos.json
OUT ?= ./output.json
ENDY ?= 2026
ENDM ?= 01
ENDD ?= 28
END = $(ENDY)-$(ENDM)-$(ENDD)
#Sources
SRCS = extensorDatos.py 

.PHONY: run help clean requirements

requirements:
	@echo "Instalando dependencias..."
	@pip install numpy

help:
	@echo "-----------------------"
	@echo "Targets disponibles:"
	@echo "  run    - Ejecuta el programa con los datos especificados."
	@echo "  clean  - Limpia los archivos generados por el programa."
	@echo "  help   - Muestra esta ayuda."
	@echo "-----------------------"
	@echo "Variables:"
	@echo "  DATOS - Archivo de datos a procesar (default: ./datos.json)"
	@echo "  OUT   - Archivo de salida (default: ./output.json)"

run: $(SRCS)
	@echo "Ejecutando el programa..."
	@$(CC) $(SRCS) $(DATOS) $(OUT) $(END)

clean:
	@echo "Limpiando archivos generados..."
	@rm -f $(OUT)
	@echo "Archivos generados eliminados."