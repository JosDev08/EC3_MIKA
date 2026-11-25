# Añadir el siguiente comando en CMD para descargar las librerias

# python -m pip install pandas

import unicodedata

# Con "Path" se puede trabar con rutas de archivos y carpetas
from pathlib import Path 

import pandas as pd # Sirve para trabajar con tablas de datos

# Averigua en qué carpeta está el archivo "codigo.py" y la guarda en la variable "CARPETA".
CARPETA = Path(__file__).resolve().parent

# Ruta CSV
ARCHIVO_CSV = CARPETA / "reto2_finanzas_transacciones_billetera.csv"

df = pd.read_csv(ARCHIVO_CSV)
# print(df.head())

# print(f"Primer vistazo, {df.shape[0]}")

# Quitamos filas repetidas
df_limpio = df.drop_duplicates() # ---> Elimina filas repetidas y lo guardamos en una varible
# print(f"Segundo vistazo, {df_limpio.shape[0]}") # ---> "shape" da (filas y columnas) con shape[0] solo me da filas

# Asignacion de nombres a distritos 
df_limpio["distrito"] = df_limpio["distrito"].str.strip().str.upper() # ---> "strip()" quita los espacion de los bordes, mientras que str hace que trate cada linea como texto.

# Comprobacion de distritos

# print(df_limpio["distrito"].unique()) # ---> "unique()" da lista de valores distintos que quedaron

# Entrada 

def entrada(texto):
    """
    Funciona como una maquina, donde entra una pregunta y la procesamos de manera que sale una pregunta mas limpia de la inicialmente entro.
    """
    texto = texto.lower().strip()  # Quitamos espacios y pasamos a minusculas.
    texto = unicodedata.normalize("NFD", texto) # Quitamos acentos "NFD" separa las letras de las tildes.
    return "".join(c for c in texto if unicodedata.category(c) != "Mn") # Rearma el texto, pero sin tildes.

# print(entrada("¿Que operación tiene más fraudes?"))

# EL vocabulario del bot, creamos un diccionario con la siguiente logica:
# palabra del usuario ---> columna real en la tabla

CLASES = {
    "distrito" : "distrito",
    "negocio" : "tipo_negocio",
    "operacion" : "tipo_operacion",
    "hora" : "hora",
    "fecha" : "fecha",
    "dispositivo" : "dispositivo_nuevo"
}

MEDIDA = {
    "fraude" : ("es_fraude", "sum"),
    "monto" : ("monto_soles", "mean"),
    "antiguedad" : ("antiguedad_cuenta_dias", "mean")
    }

# print(MEDIDA["fraude"])

def busqueda(texto, diccionario):
    """
    Recorre un diccionario y devuelve el valor de la primera palabra que aparezca en la pregunta
    """
    for palabra, valor in diccionario.items(): # ---> "items()" hagarra tanto la clave como el valor
        if palabra in texto: # Si la palabra aparece dentro de la frase entonces devuelve "valor"
            return valor
    return None


# Como se llama cada calculo cuando lo mostramos
NOMBRE_CALCULO = {
    "sum" : "Total",
    "mean" : "Promedio"
}

# Como se llama cada columna medida cuando la mostramos
NOMBRE_MEDIDA = {
    "es_fraude" : "fraudes",
    "monto_soles" : "monto (S/)",
    "antiguedad_cuenta_dias" : "antiguedad (dias)"
}

def respuesta(pregunta):    
    texto = entrada(pregunta)
    columna = busqueda(texto, CLASES)
    medida = busqueda(texto, MEDIDA)
    
    # Si no se detecta bien lo que se solicita entonces sugerir algo.
    
    if medida is None:
        return "No entendi, prueba con: fraudes por distrito"

    col_medida, calculo = medida
    nombre_calculo = NOMBRE_CALCULO[calculo]
    nombre_medida = NOMBRE_MEDIDA[col_medida]

    if columna is None:
        valor = round(df_limpio[col_medida].agg(calculo), 2)
        return f"{nombre_calculo} de {nombre_medida}: {valor}"

    # Si la pregunta trae "menos", ordenamos de menor a mayor
    de_menor_a_mayor = "menos" in texto

    resultado = (
        df_limpio.groupby(columna)[col_medida]
        .agg(calculo)
        .sort_values(ascending = de_menor_a_mayor)
        .round(2)
    )

    lista = ""
    for nombre, valor in resultado.items():
        lista += f"  {nombre}: {valor}\n"

    nombre_clase = columna.replace("_", " ")
    orden = "de menor a mayor" if de_menor_a_mayor else "de mayor a menor"
    return f"{nombre_calculo} de {nombre_medida} por {nombre_clase} ({orden}):\n{lista}"
        
pregunta = entrada("¿Que distrito tiene mas fraudes?") # Esto hace que el texto quede limpio
#pregunta = entrada("A?")
# print(pregunta)
# print(busqueda(pregunta, CLASES)) # Procesa la pregunta
# print(busqueda(pregunta, MEDIDA)) # Procesa la pregunta

# OPERACIONES 
## Aqrupamos las filas por distrito
# Este es un primer modelo : "resultado = df_limpio.groupby("distrito")["es_fraude"].sum() # "sum()" suma esa columna, con esto contamos el numero de fraudes."

resultado = (
    df_limpio.groupby("distrito")["es_fraude"]
    .sum()
    .sort_values(ascending = False) # ordena de mayor a menor, si "ascendig = True" seria alrevez
)
#promedio = (
    #df_limpio.groupby("distrito")["es_fraude"]
    #.mean()
    #)
# print(f"a : {round((promedio * 100),2)} %")
#print(resultado)

df_limpio = df_limpio.copy()
df_limpio["monto_soles"] = df_limpio["monto_soles"].fillna(df_limpio["monto_soles"].median())
df_limpio["antiguedad_cuenta_dias"] = df_limpio["antiguedad_cuenta_dias"].fillna(df_limpio["antiguedad_cuenta_dias"].median())
for col in ["tipo_operacion", "tipo_negocio", "dispositivo_nuevo"]:
    df_limpio[col] = df_limpio[col].str.strip().str.upper()

train = df_limpio.sample(frac=0.80, random_state=42)
test = df_limpio.drop(train.index).copy()

def entrenar_modelo(datos_train):
    return {
        "umbral_monto": round(datos_train["monto_soles"].quantile(0.92), 2),
        "umbral_antiguedad": 30,
        "umbral_hora": 5,
    }

parametros_ia = entrenar_modelo(train)

def predecir_fraude(fila, params):
    senales = 0
    if fila["antiguedad_cuenta_dias"] < params["umbral_antiguedad"]:
        senales += 1
    if fila["hora"] <= params["umbral_hora"]:
        senales += 1
    if fila["dispositivo_nuevo"] == "SÍ":
        senales += 1

    if fila["monto_soles"] > params["umbral_monto"] or senales >= 2:
        return 1
    return 0

test["prediccion"] = test.apply(lambda f: predecir_fraude(f, parametros_ia), axis=1)

aciertos = (test["es_fraude"] == test["prediccion"]).sum()
accuracy = round((aciertos / len(test)) * 100, 2)

vp = ((test["es_fraude"] == 1) & (test["prediccion"] == 1)).sum()
fraudes_reales = (test["es_fraude"] == 1).sum()
recall = round((vp / fraudes_reales) * 100, 2)

matriz_confusion = pd.crosstab(
    test["es_fraude"], test["prediccion"],
    rownames=["Real"], colnames=["Predicho"]
)

test["falsa_alarma"] = (test["es_fraude"] == 0) & (test["prediccion"] == 1)
legitimas = test[test["es_fraude"] == 0]
tabla_etica = legitimas.groupby("distrito")["falsa_alarma"].agg(
    legitimas="count",
    falsas_alarmas="sum",
    tasa_error="mean"
)
tabla_etica["tasa_error_%"] = (tabla_etica["tasa_error"] * 100).round(2)

df_limpio["fraude_predicho"] = df_limpio.apply(lambda f: predecir_fraude(f, parametros_ia), axis=1)

def accion_etica(fila):
    if fila["fraude_predicho"] == 1 and (fila["monto_soles"] >= 200 or fila["dispositivo_nuevo"] == "SÍ"):
        return "BLOQUEO PREVENTIVO + LLAMADA"
    elif fila["fraude_predicho"] == 1:
        return "VERIFICACION SMS (OTP)"
    else:
        return "APROBADA"

df_limpio["accion_sistema"] = df_limpio.apply(accion_etica, axis=1)

CLASES["accion"] = "accion_sistema"
MEDIDA["predicho"] = ("fraude_predicho", "sum")
NOMBRE_MEDIDA["fraude_predicho"] = "fraudes predichos por IA"

# 9. Imprimimos los resultados EC3 al iniciar el programa
print("=== RESULTADOS DEL MODELO DE CLASIFICACION (EC3) ===")
print(f"Division: {len(train)} train (80%) | {len(test)} test (20%)")
print(f"Accuracy: {accuracy}% ({aciertos} de {len(test)} aciertos)")
print(f"Recall:   {recall}% ({vp} de {fraudes_reales} fraudes reales detectados)")
print("\nMatriz de Confusion:")
print(matriz_confusion)
print("\nTabla de Errores por Distrito (Etica Algoritmica):")
print(tabla_etica[["legitimas", "falsas_alarmas", "tasa_error_%"]])
print("====================================================\n")

if __name__ == "__main__":
    print("Ingrese su consulta. Escriba 'salir' para finalizar las operaciones")
    while True: 
        pregunta = input("Pregunta: ")
        if entrada(pregunta) == "salir":
            break
        print(respuesta(pregunta), "\n")