import json
import random
from datetime import datetime, timedelta
import argparse
import numpy as np
from scipy import stats

def generar_datos_extendidos(datos_originales, fecha_final_str):
    """
    Genera datos meteorológicos extendidos a partir de un conjunto de datos originales hasta una fecha específica.

    Parámetros:
        datos_originales (list): Lista de diccionarios con los datos meteorológicos originales.
        fecha_final_str (str): Fecha final para la extensión en formato "YYYY-MM-DD".

    Retorna:
        list: Lista de diccionarios con los datos originales y los datos extendidos.
    """
    # Convertir fechas a objetos datetime
    fecha_ultima = datetime.strptime(datos_originales[-1]['fecha'], "%Y-%m-%d")
    fecha_final = datetime.strptime(fecha_final_str, "%Y-%m-%d")
    
    # Verificar que la fecha final sea posterior a la última fecha en los datos
    if fecha_final <= fecha_ultima:
        print("La fecha final debe ser posterior a la última fecha en los datos.")
        return datos_originales
    
    # Preparar datos para análisis de tendencias
    fechas = [datetime.strptime(d['fecha'], "%Y-%m-%d") for d in datos_originales]
    dias_del_ano = [d.timetuple().tm_yday for d in fechas]
    
    # Extraer valores históricos por día del año
    datos_por_dia = {}
    for i, dia in enumerate(dias_del_ano):
        if dia not in datos_por_dia:
            datos_por_dia[dia] = []
        datos_por_dia[dia].append(datos_originales[i])
    
    # Calcular estadísticas por día del año
    estadisticas_por_dia = {}
    for dia in datos_por_dia:
        datos_dia = datos_por_dia[dia]
        
        # Para precipitación (considerando días con y sin lluvia)
        prec_values = []
        for d in datos_dia:
            prec = d['prec']
            if isinstance(prec, str):
                prec = prec.replace('Ip', '0.1').replace(',', '.')
            prec_values.append(float(prec))
        
        prob_lluvia = sum(1 for p in prec_values if p > 0) / len(prec_values)
        
        # Para temperaturas y velocidad del viento
        tmed_values = [float(d['tmed'].replace(',', '.')) for d in datos_dia]
        tmax_values = [float(d['tmax'].replace(',', '.')) for d in datos_dia]
        tmin_values = [float(d['tmin'].replace(',', '.')) for d in datos_dia]
        velmedia_values = [float(d['velmedia'].replace(',', '.')) for d in datos_dia]
        
        # Guardar estadísticas calculadas
        estadisticas_por_dia[dia] = {
            'prob_lluvia': prob_lluvia,
            'prec_mean': np.mean(prec_values),
            'prec_std': np.std(prec_values),
            'tmed_mean': np.mean(tmed_values),
            'tmed_std': np.std(tmed_values),
            'tmax_mean': np.mean(tmax_values),
            'tmax_std': np.std(tmax_values),
            'tmin_mean': np.mean(tmin_values),
            'tmin_std': np.std(tmin_values),
            'velmedia_mean': np.mean(velmedia_values),
            'velmedia_std': np.std(velmedia_values)
        }
    
    # Generar datos extendidos
    datos_extendidos = datos_originales.copy()
    current_date = fecha_ultima + timedelta(days=1)
    
    while current_date <= fecha_final:
        dia_del_ano = current_date.timetuple().tm_yday
        
        # Obtener estadísticas para este día del año
        if dia_del_ano in estadisticas_por_dia:
            stats_dia = estadisticas_por_dia[dia_del_ano]
        else:
            # Si no hay datos para este día, usar el promedio de días cercanos
            dias_cercanos = [d for d in estadisticas_por_dia.keys() 
                            if abs(d - dia_del_ano) <= 7 or abs(d - dia_del_ano + 365) <= 7]
            if not dias_cercanos:
                dias_cercanos = list(estadisticas_por_dia.keys())
            
            stats_dia = {
                'prob_lluvia': np.mean([estadisticas_por_dia[d]['prob_lluvia'] for d in dias_cercanos]),
                'prec_mean': np.mean([estadisticas_por_dia[d]['prec_mean'] for d in dias_cercanos]),
                'prec_std': np.mean([estadisticas_por_dia[d]['prec_std'] for d in dias_cercanos]),
                'tmed_mean': np.mean([estadisticas_por_dia[d]['tmed_mean'] for d in dias_cercanos]),
                'tmed_std': np.mean([estadisticas_por_dia[d]['tmed_std'] for d in dias_cercanos]),
                'tmax_mean': np.mean([estadisticas_por_dia[d]['tmax_mean'] for d in dias_cercanos]),
                'tmax_std': np.mean([estadisticas_por_dia[d]['tmax_std'] for d in dias_cercanos]),
                'tmin_mean': np.mean([estadisticas_por_dia[d]['tmin_mean'] for d in dias_cercanos]),
                'tmin_std': np.mean([estadisticas_por_dia[d]['tmin_std'] for d in dias_cercanos]),
                'velmedia_mean': np.mean([estadisticas_por_dia[d]['velmedia_mean'] for d in dias_cercanos]),
                'velmedia_std': np.mean([estadisticas_por_dia[d]['velmedia_std'] for d in dias_cercanos])
            }
        
        # Generar precipitación (distribución mixta)
        if random.random() < stats_dia['prob_lluvia']:
            prec = max(0, np.random.normal(stats_dia['prec_mean'], stats_dia['prec_std']))
            prec = round(max(0, prec), 1)  # Redondear a 1 decimal y asegurar que no sea negativo
            if prec < 0.1 and random.random() < 0.1:  # 10% de probabilidad de "Ip"
                prec = "Ip"
        else:
            prec = 0.0
        
        # Generar temperaturas y velocidad del viento con distribución normal
        tmed = round(np.random.normal(stats_dia['tmed_mean'], stats_dia['tmed_std']), 1)
        tmax = round(np.random.normal(stats_dia['tmax_mean'], stats_dia['tmax_std']), 1)
        tmin = round(np.random.normal(stats_dia['tmin_mean'], stats_dia['tmin_std']), 1)
        velmedia = round(max(0, np.random.normal(stats_dia['velmedia_mean'], stats_dia['velmedia_std'])), 1)
        
        # Asegurar que tmax >= tmed >= tmin
        tmed = min(tmax, max(tmin, tmed))
        
        # Crear nuevo registro
        nuevo_dato = {
            "fecha": current_date.strftime("%Y-%m-%d"),
            "indicativo": datos_originales[0]["indicativo"],
            "nombre": datos_originales[0]["nombre"],
            "provincia": datos_originales[0]["provincia"],
            "altitud": datos_originales[0]["altitud"],
            "tmed": str(tmed).replace('.', ','),
            "prec": str(prec).replace('.', ','),
            "tmin": str(tmin).replace('.', ','),
            "horatmin": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            "tmax": str(tmax).replace('.', ','),
            "horatmax": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            "dir": str(random.randint(0, 36)),
            "velmedia": str(velmedia).replace('.', ','),
            "racha": str(round(velmedia * random.uniform(1.5, 3.0), 1)).replace('.', ','),
            "horaracha": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            "sol": str(round(random.uniform(0, 12), 1)).replace('.', ','),
            "presMax": str(round(random.uniform(990, 1030), 1)).replace('.', ','),
            "horaPresMax": f"{random.randint(0, 23):02d}",
            "presMin": str(round(random.uniform(990, 1030), 1)).replace('.', ','),
            "horaPresMin": f"{random.randint(0, 23):02d}",
            "hrMedia": str(random.randint(50, 90)),
            "hrMax": str(random.randint(70, 99)),
            "horaHrMax": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            "hrMin": str(random.randint(40, 80)),
            "horaHrMin": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
        }
        
        datos_extendidos.append(nuevo_dato)
        current_date += timedelta(days=1)
    
    return datos_extendidos

def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description='Extender datos meteorológicos hasta una fecha específica.')
    parser.add_argument('input_file', help='Archivo JSON de entrada con los datos meteorológicos')
    parser.add_argument('output_file', help='Archivo JSON de salida con los datos extendidos')
    parser.add_argument('fecha_final', help='Fecha final para la extensión (formato YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Leer el archivo de entrada con codificación ISO-8859-1 para manejar caracteres españoles
    with open(args.input_file, 'r', encoding='iso-8859-1') as f:
        datos = json.load(f)
    
    # Generar datos extendidos
    datos_extendidos = generar_datos_extendidos(datos, args.fecha_final)
    
    # Escribir el archivo de salida
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(datos_extendidos, f, indent=2, ensure_ascii=False)
    
    print(f"Datos extendidos hasta {args.fecha_final} y guardados en {args.output_file}")

if __name__ == "__main__":
    main()