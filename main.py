
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# PRACTICA 3: LIMPIEZA AVANZADA DE DATOS
# ==========================================

# 1. GENERACION DE DATOS CON RUIDO
data = {
    "edad": [
        25, 30, np.nan, 45, 22,
        120, 35, 28, np.nan, 40
    ],
    "ingreso_mensual": [
        15000, 22000, 18000, np.nan, 12000,
        950000, 25000, 21000, 19000, np.nan
    ],
    "scoring_credito": [
        650, 700, 680, 710, 590,
        800, 670, 640, 620, 730
    ]
}

df_original = pd.DataFrame(data)
df_dirty = df_original.copy()

print("\n=== DATASET ORIGINAL ===")
print(df_original)

print("\n=== VALORES NULOS ORIGINALES ===")
print(df_original.isnull().sum())

print("\n=== INFORMACION DEL DATASET ===")
print(f"Filas: {df_original.shape[0]}")
print(f"Columnas: {df_original.shape[1]}")


# 2. DETECCION DE OUTLIERS MEDIANTE IQR
print("\n=== DETECCION DE OUTLIERS (IQR) ===")

Q1 = df_dirty["ingreso_mensual"].quantile(0.25)
Q3 = df_dirty["ingreso_mensual"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Q1: ${Q1:,.2f}")
print(f"Q3: ${Q3:,.2f}")
print(f"IQR: ${IQR:,.2f}")
print(f"Limite inferior: ${limite_inferior:,.2f}")
print(f"Limite superior: ${limite_superior:,.2f}")

# Identificar ingresos fuera de los limites
outliers_ingreso = (
    (df_dirty["ingreso_mensual"] < limite_inferior) |
    (df_dirty["ingreso_mensual"] > limite_superior)
)

print("\nIngresos atipicos encontrados:")
print(df_dirty.loc[outliers_ingreso])

# Reemplazar outliers por NaN
df_dirty.loc[
    outliers_ingreso, "ingreso_mensual"
] = np.nan

# Corregir edades superiores a 100
edades_invalidas = df_dirty["edad"] > 100

print("\nEdades invalidas encontradas:")
print(df_dirty.loc[edades_invalidas])

df_dirty.loc[edades_invalidas, "edad"] = np.nan

print("\n=== DATASET DESPUES DE DETECTAR ERRORES ===")
print(df_dirty)

print("\n=== VALORES NULOS POR IMPUTAR ===")
print(df_dirty.isnull().sum())


# 3. IMPUTACION MEDIANTE KNN (K=2)
print("\n=== IMPUTACION KNN CON K=2 ===")

imputer_knn = KNNImputer(n_neighbors=2)

df_clean_array = imputer_knn.fit_transform(df_dirty)

df_clean = pd.DataFrame(
    df_clean_array,
    columns=df_dirty.columns,
    index=df_dirty.index
)

print(df_clean)

print("\nValores nulos despues de KNN:")
print(df_clean.isnull().sum())


# 4. COMPARACION CON OTROS VALORES DE K
print("\n=== COMPARACION DE IMPUTACIONES ===")

resultados = {}

for k in [2, 3, 5]:
    imputer = KNNImputer(n_neighbors=k)

    resultado = pd.DataFrame(
        imputer.fit_transform(df_dirty),
        columns=df_dirty.columns,
        index=df_dirty.index
    )

    resultados[f"KNN_k{k}"] = resultado

# Imputacion tradicional por media
imputer_media = SimpleImputer(strategy="mean")

df_media = pd.DataFrame(
    imputer_media.fit_transform(df_dirty),
    columns=df_dirty.columns,
    index=df_dirty.index
)

resultados["Media"] = df_media

# Imputacion tradicional por mediana
imputer_mediana = SimpleImputer(strategy="median")

df_mediana = pd.DataFrame(
    imputer_mediana.fit_transform(df_dirty),
    columns=df_dirty.columns,
    index=df_dirty.index
)

resultados["Mediana"] = df_mediana

# Comparar solamente los valores imputados
comparacion = []

for fila, columna in zip(
    *np.where(df_dirty.isna().to_numpy())
):
    nombre_columna = df_dirty.columns[columna]

    registro = {
        "fila": fila,
        "columna": nombre_columna
    }

    for metodo, resultado in resultados.items():
        registro[metodo] = resultado.iloc[fila, columna]

    comparacion.append(registro)

df_comparacion = pd.DataFrame(comparacion)

print(df_comparacion.to_string(index=False))


# 5. VISUALIZACION: ANTES Y DESPUES
print("\n=== GENERANDO GRAFICAS ===")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Boxplot original
axes[0].boxplot(
    df_original["ingreso_mensual"].dropna()
)
axes[0].set_title("Antes de la limpieza")
axes[0].set_ylabel("Ingreso mensual ($)")
axes[0].grid(axis="y", alpha=0.3)

# Boxplot despues de KNN
axes[1].boxplot(
    df_clean["ingreso_mensual"].dropna()
)
axes[1].set_title("Despues de la limpieza")
axes[1].set_ylabel("Ingreso mensual ($)")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Comparacion de Outliers - Metodo IQR y KNN")
fig.tight_layout()

plt.savefig(
    "comparacion_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Grafica guardada: comparacion_boxplots.png")


# 6. EXPORTAR RESULTADOS
df_clean.to_csv(
    "dataset_limpio.csv",
    index=False
)

df_comparacion.to_csv(
    "comparacion_imputaciones.csv",
    index=False
)

print("\nArchivos CSV generados correctamente.")


# 7. VALIDACION FINAL
print("\n=== VALIDACION FINAL ===")

assert df_clean.shape == df_original.shape
assert not df_clean.isnull().any().any()

print("Dimensiones conservadas correctamente.")
print("No existen valores nulos.")
print("Limpieza e imputacion completadas.")

print("\n=== PRACTICA 3 FINALIZADA ===")
