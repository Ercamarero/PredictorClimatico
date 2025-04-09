# Predictor Climático - Proyecto PAM

Este `README` fue escrito para la compilación, modificación en caso de querer experimentar y usar el proyecto; es por ello que explicaremos detalles acerca del código y el `Makefile`.
Junto con este boletín se adjunta un `Makefile` que automatiza todos los comandos que deberíais de necesitar para la comprensión y utilización del entorno.

## Jerarquía de ficheros

- `Arima` Conjunto de ficheros que realizan la definición y realización de los cálculos tanto para el modelo de cálculo de ARIMA como una regresión lineal simple para la comparativa.
- `JSONParserHibrido` Fichero principal: donde se encuentra el main, contiene toda la estructuración del parsing y uso de los datos; es el módulo principal.
- `cJSON` Librería externa de la que dependemos para la lectura del JSON.

## Uso

Se incluyen diferentes targets en el Makefile para la compilación y ejecución.

Modo de uso:

```bash
# 1. En caso de desear modificar cualquier variable realiza lo siguiente:
# MODIFICAR EL INPUT
export DATOS=<FICHERO.JSON> # Lugar del que extraemos los datos
# MODIFICAR EL NUMERO DE PROCESOS MPI
export NP=<NumeroProcesosMPI>
# MODIFICAR LOS PARAMETROS OBTENIDOS
export CAMPOS=<"NuevosCampos"> #Contenidos entre "" en caso de querer añadir mas de uno
#EJEMPLO:
# export CAMPOS="tmax tmed prec"
# MODIFICAR EL NUMERO DE HILOS OMP
export NHILOS=<Cantidad de hilos>

# 2. Compilación del entorno
make all

# 3. Ejecución del entorno con los parametros indicados
make run

# 4. En caso de perdida siempre se puede ejecutar 
make help

# 5. Limpieza del entorno 
make clean
```

## Posibles modificaciones para el usuario

- Debido a la flexibilidad de la implementación del modelo de ARIMA, me vi en la situación de añadir una estructura de modelos que nos permitiese cargar parámetros para ARIMA, pudiendo así modificar el comportamiento de nuestro predictor.

```C
     // ARIMA Models
        //Estructura que nos permite alterar los modelos que usara nuestro sistema de prediccion añadiendo diversos modelos 
        ARIMA_Model modelos_arima[] = {
            {1, 0, 0},  //MODELO GENERICO PARA DATOS ESTACIONARIOS CLIMATICOS 
        };
```

A la hora de querer ajustar nuestro modelo para los distintos usos que tiene este gran modelo matematico podemos introducir entre parentesis la tripleta de valores `{p, d, q}` los cuales se explicaran posteriormente; introducir nuevas tripletas de valores hara que los campos extraidos del `JSON` recibiran ese trato.
Todos los modelos que se añadan a la estructura de modelos se ejecutaran para los distintos parametros que se hayan pedido de busqueda por lo que nos permite ver como afecta cada uno al resultado.

## Significado de los valores

- Yendo en orden de aparición:
  
1. **Componente AR (p) - Autorregresivo:**

   - **Qué representa:** Cuántos valores pasados influyen en el valor actual.  
   - **Efecto:**  
     - `p=1`: El valor actual depende solo del valor inmediatamente anterior.  
     - `p=2`: Considera los dos últimos valores.  
     - Valores altos (`p>3`): Capturan patrones complejos pero pueden llevar a sobreajuste..

2. **Componente I (d) - Integración:**

   - **Qué representa:** Número de veces que se diferencia la serie para hacerla estacionaria.  
   - **Efecto:**  
     - `d=0`: No se aplica diferenciación (serie ya estacionaria).  
     - `d=1`: Remueve tendencia lineal.  
     - `d=2`: Remueve tendencia cuadrática.

3. **Componente MA (q) - Media Móvil:**

   - **Qué representa:** Cuántos errores pasados influyen en el valor actual.  
   - **Efecto:**  
     - `q=1`: Considera el error de la predicción anterior.  
     - `q=2`: Considera los dos últimos errores.  
     - Valores altos (`p>3`): Modela shocks temporales pero puede volverse ruidoso.

## Principales valores

- `tmax` Temperatura máxima.
- `tmed` Temperatura media.
- `tmin` Temperatura mínima.
- `velmedia` Velocidad del viento media.
- `prec` Precipitaciones.
- `sol` Incidencia ultravioleta.
- `presmax` Presión atmosférica máxima.
- `presmin` Presión atmosférica máxima.

## Referencias

- [Libreria cJSON](https://github.com/DaveGamble/cJSON/tree/master)
