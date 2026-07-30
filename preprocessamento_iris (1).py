"""
Aplicacao pratica - Pre-processamento de Dados
Projeto Final - Banco de Dados II - UFU

Este script demonstra, de forma didatica, um pipeline completo de
pre-processamento de dados aplicado sobre o dataset publico Iris
(Fisher, 1936; disponibilizado pela biblioteca scikit-learn).

Para tornar a demonstracao realista, o dataset original (que e limpo)
recebe uma "sujeira sintetica" controlada: valores ausentes, registros
duplicados, outliers e uma variavel categorica com inconsistencias de
escrita. Em seguida, aplicam-se as etapas classicas do pre-processamento:
limpeza, transformacao (normalizacao/padronizacao), codificacao de
variaveis categoricas e validacao final, terminando com a comparacao do
desempenho de um classificador antes e depois do pre-processamento.

Autor: [Nome dos integrantes do grupo]
Disciplina: Banco de Dados II - Prof. Igor da Penha Natal
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ---------------------------------------------------------------------
# 1. CARGA DOS DADOS E INJECAO DE "SUJEIRA" SINTETICA (para fins didaticos)
# ---------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
df["species"] = df["target"].map(species_map)

# 1.1 Inconsistencias de escrita na variavel categorica (categoria "suja")
variacoes = {
    "setosa": ["setosa", "Setosa", "SETOSA ", " setosa"],
    "versicolor": ["versicolor", "Versicolor", "VERSICOLOR", "versicolor "],
    "virginica": ["virginica", "Virginica", "VIRGINICA ", " virginica"],
}
df["species_raw"] = df["species"].apply(lambda s: rng.choice(variacoes[s]))

# 1.2 Duplicatas propositais (5% dos registros)
dup_idx = rng.choice(df.index, size=int(0.05 * len(df)), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 1.3 Valores ausentes propositais (~8% em duas colunas numericas)
for col in ["sepal_length", "petal_width"]:
    miss_idx = rng.choice(df.index, size=int(0.08 * len(df)), replace=False)
    df.loc[miss_idx, col] = np.nan

# 1.4 Outliers propositais (multiplicando alguns valores por um fator extremo)
out_idx = rng.choice(df.index, size=6, replace=False)
df.loc[out_idx, "sepal_width"] = df.loc[out_idx, "sepal_width"] * rng.uniform(3, 4, size=6)

df_raw = df.copy()

print("=" * 70)
print("ETAPA 0 - PANORAMA DOS DADOS BRUTOS")
print("=" * 70)
print(f"Dimensoes: {df_raw.shape[0]} linhas x {df_raw.shape[1]} colunas")
print("\nValores ausentes por coluna:")
print(df_raw.isnull().sum())
print(f"\nRegistros duplicados (excluindo 'species_raw'/'target'): "
      f"{df_raw.drop(columns=['species_raw']).duplicated().sum()}")
print("\nExemplos de inconsistencia categorica em 'species_raw':")
print(df_raw["species_raw"].unique()[:8])

# ---------------------------------------------------------------------
# 2. LIMPEZA DOS DADOS
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("ETAPA 1 - LIMPEZA")
print("=" * 70)

# 2.1 Padronizacao de strings (remover espacos e uniformizar caixa)
df["species_clean"] = df["species_raw"].str.strip().str.lower()
print("Categorias apos padronizacao de texto:", sorted(df["species_clean"].unique()))

# 2.2 Remocao de duplicatas exatas (considerando as colunas de features + rotulo)
antes = len(df)
df = df.drop_duplicates(subset=["sepal_length", "sepal_width", "petal_length",
                                 "petal_width", "species_clean"])
print(f"Duplicatas removidas: {antes - len(df)} (de {antes} para {len(df)} linhas)")

# 2.3 Deteccao e tratamento de outliers via IQR (na coluna sepal_width)
q1, q3 = df["sepal_width"].quantile([0.25, 0.75])
iqr = q3 - q1
limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["sepal_width"] < limite_inferior) | (df["sepal_width"] > limite_superior)]
print(f"Outliers detectados via IQR em 'sepal_width': {len(outliers)}")
df.loc[outliers.index, "sepal_width"] = np.nan  # tratados como ausentes para posterior imputacao

# 2.4 Tratamento de valores ausentes (imputacao pela mediana, por especie)
print("Valores ausentes antes da imputacao:")
print(df[["sepal_length", "sepal_width", "petal_width"]].isnull().sum())

for col in ["sepal_length", "sepal_width", "petal_width"]:
    df[col] = df.groupby("species_clean")[col].transform(lambda s: s.fillna(s.median()))

print("Valores ausentes apos a imputacao (mediana por especie):")
print(df[["sepal_length", "sepal_width", "petal_width"]].isnull().sum())

# ---------------------------------------------------------------------
# 3. TRANSFORMACAO: NORMALIZACAO/PADRONIZACAO E CODIFICACAO
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("ETAPA 2 - TRANSFORMACAO")
print("=" * 70)

# Feature adicional com magnitude bem maior que as demais (area da petala em
# mm2), simulando o cenario classico em que uma variavel de escala grande
# (ex.: salario) domina o calculo de distancia sobre variaveis de escala
# pequena (ex.: idade) caso nao haja normalizacao.
df["petal_area_mm2"] = df["petal_length"] * df["petal_width"] * 100

features_num = ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_area_mm2"]
X = df[features_num]
y = df["species_clean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# --- Baseline SEM pre-processamento (apenas para efeito comparativo) ---
# KNN foi escolhido por ser um algoritmo baseado em distancia, portanto
# sensivel a escala das variaveis - o que evidencia o efeito da normalizacao.
modelo_sem = KNeighborsClassifier(n_neighbors=5)
modelo_sem.fit(X_train, y_train)
pred_sem = modelo_sem.predict(X_test)
acc_sem = accuracy_score(y_test, pred_sem)
f1_sem = f1_score(y_test, pred_sem, average="macro")

# --- Pipeline COM pre-processamento (padronizacao Z-score) ---
# IMPORTANTE: o scaler e "ajustado" (fit) somente no conjunto de treino,
# evitando vazamento de dados (data leakage) para o conjunto de teste.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5)),
])
pipeline.fit(X_train, y_train)
pred_com = pipeline.predict(X_test)
acc_com = accuracy_score(y_test, pred_com)
f1_com = f1_score(y_test, pred_com, average="macro")

print(f"Acuracia SEM pre-processamento : {acc_sem:.4f} | F1-macro: {f1_sem:.4f}")
print(f"Acuracia COM pre-processamento : {acc_com:.4f} | F1-macro: {f1_com:.4f}")

# 3.1 Exemplo de codificacao categorica (One-Hot) para fins ilustrativos
encoder = OneHotEncoder(sparse_output=False)
species_encoded = encoder.fit_transform(df[["species_clean"]])
encoded_df = pd.DataFrame(species_encoded, columns=encoder.get_feature_names_out())
print("\nExemplo de One-Hot Encoding da variavel 'species_clean' (5 primeiras linhas):")
print(encoded_df.head())

# ---------------------------------------------------------------------
# 4. VALIDACAO FINAL E EXPORTACAO
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("ETAPA 3 - VALIDACAO FINAL")
print("=" * 70)
print(f"Linhas: bruto={len(df_raw)} -> processado={len(df)}")
print(f"Valores ausentes restantes: {df[features_num].isnull().sum().sum()}")
print(f"Duplicatas restantes: {df.drop(columns=['species_raw']).duplicated().sum()}")

df_final = df[features_num + ["species_clean"]].rename(columns={"species_clean": "species"})
df_final.to_csv("iris_preprocessado.csv", index=False)
print("\nArquivo 'iris_preprocessado.csv' exportado com sucesso.")

print("\n" + "=" * 70)
print("RESUMO COMPARATIVO")
print("=" * 70)
print(f"{'Metrica':<20}{'Sem pre-proc.':<18}{'Com pre-proc.':<18}")
print(f"{'Acuracia':<20}{acc_sem:<18.4f}{acc_com:<18.4f}")
print(f"{'F1-macro':<20}{f1_sem:<18.4f}{f1_com:<18.4f}")
