# Extensor Datos - Proyecto PAM

Este `README` fue escrito para la compilación, modificación en caso de querer experimentar y usar el proyecto; es por ello que explicaremos detalles acerca del código y el `Makefile`.
Junto con este boletín se adjunta un `Makefile` que automatiza todos los comandos que deberíais de necesitar para la comprensión y utilización del entorno.

## Jerarquía de ficheros

- `extensorDatos` Está parte del proyecto solo cuenta con un fichero el cual hará toda la funcionalidad adicional necesaria para la creacion de datos sintéticos.

## Uso

Se incluyen diferentes targets en el Makefile para la compilación y ejecución.

Modo de uso:

```bash
# 1. En caso de desear modificar cualquier variable realiza lo siguiente:
# MODIFICAR EL INPUT
export DATOS=<FICHERO.JSON> # Lugar del que extraemos los datos originales
# MODIFICAR EL NOMBRE DE LA SALIDA 
export OUT=<"NuevoNombre"> #Contenidos entre "" en caso de querer añadir mas de uno
#EJEMPLO:
# MODIFICAR EL AÑO DE FIN DE PRODUCCION DE DATOS
export ENDY=<Cantidad de hilos>
# MODIFICAR EL MES DE FIN DE PRODUCCION DE DATOS
export ENDM=<Cantidad de hilos>
# MODIFICAR EL DÍA DE FIN DE PRODUCCION DE DATOS
export ENDD=<Cantidad de hilos>

# 2. Ejecución del entorno con los parametros indicados
make run

# 3. En caso de perdida siempre se puede ejecutar 
make help

# 4. Limpieza del entorno 
make clean

# 5. Descarga de los requisitos
make requirements 
```

Este módulo depende de Numpy es por ello que en caso de no tenerlo instalado en tu maquina puedes usar el `make requirements` para descargar dicha librería.

## Salida

La salida de este script es un fichero tipo JSON que mantiene el esquema de aquellos que utiliza la AEMET en su API.
