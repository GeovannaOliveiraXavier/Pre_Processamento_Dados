

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

Autores: Alberto Nogueira Neto, Ana Carolina Gomes Lisbôa, Geovanna de Oliveira Xavier, Isabelle Martins Cavalcante
Disciplina: Banco de Dados II - Prof. Igor da Penha Natal

Aplicação no caso em que o usuário não tem um dataset ou o programa não conseguiu acessar o
arquivo:
"""

from typing import List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configurações Estéticas Globais
RANDOM_STATE: int = 42
sns.set_theme(style="whitegrid", palette="deep")

CAMINHO_DO_SEU_CSV = r"/Users/mariaritagomes/Downloads/iris_preprocessado.csv"

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa valores ausentes utilizando a mediana agrupada por classe."""
    def __init__(self, group_col: str, target_cols: List[str]):
        self.group_col = group_col
        self.target_cols = target_cols
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.target_cols:
            self.medians_[col] = X.groupby(self.group_col)[col].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.target_cols:
            group_medians = X_out[self.group_col].map(self.medians_[col])
            X_out[col] = X_out[col].fillna(group_medians)
            X_out[col] = X_out[col].fillna(X_out[col].median())
        return X_out


class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """Substitui outliers por NaN usando a regra do Amplitude Interquartílica (IQR)."""
    def __init__(self, target_col: str, factor: float = 1.5):
        self.target_col = target_col
        self.factor = factor

    def fit(self, X: pd.DataFrame, y=None):
        q1 = X[self.target_col].quantile(0.25)
        q3 = X[self.target_col].quantile(0.75)
        iqr = q3 - q1
        self.lower_bound_ = q1 - self.factor * iqr
        self.upper_bound_ = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        outliers_mask = (X_out[self.target_col] < self.lower_bound_) | (X_out[self.target_col] > self.upper_bound_)
        X_out.loc[outliers_mask, self.target_col] = np.nan
        return X_out

def load_csv_data(caminho_csv: str) -> pd.DataFrame:
    """Carrega o CSV local do Mac ou constrói o banco bruto sujo se estiver na nuvem."""
    if os.path.exists(caminho_csv):
        print(f"[+] Conectando e carregando dados brutos do arquivo no Mac:\n    -> {caminho_csv}")
        df = pd.read_csv(caminho_csv)
    else:
        print(f"[!] Aviso: O caminho local '{caminho_csv}' não foi encontrado neste ambiente.")
        print("[+] Modo Nuvem/Colab Ativado: Gerando o Banco de Dados Bruto COM SUJEIRA para o teste!")
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

        # Injeção de Sujeiras Sintéticas
        rng_temp = np.random.default_rng(RANDOM_STATE)

        # 1. Sujeira de Texto
        df["species_raw"] = df["target"].map({0: "setosa ", 1: "Versicolor", 2: "VIRGINICA"})

        # 2. Registros Duplicados (5%)
        dup_idx = rng_temp.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

        # 3. Valores Ausentes (Missing Values)
        df.loc[rng_temp.choice(df.index, 10), "sepal_length"] = np.nan
        df.loc[rng_temp.choice(df.index, 8), "petal_width"] = np.nan

        # 4. Outliers Extremos
        out_idx = rng_temp.choice(df.index, 5)
        df.loc[out_idx, "sepal_width"] = df.loc[out_idx, "sepal_width"] * 3.8

    return df

def run_data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executa o saneamento primário do dataset."""
    df_clean = df.copy()

    # 1. Normalização de Strings
    if "species_raw" in df_clean.columns:
        df_clean["species_clean"] = df_clean["species_raw"].astype(str).str.strip().str.lower()
    elif "species" in df_clean.columns:
        df_clean["species_clean"] = df_clean["species"].astype(str).str.strip().str.lower()

    # 2. Remoção de Duplicatas
    cols_check = [c for c in ["sepal_length", "sepal_width", "petal_length", "petal_width", "species_clean"] if c in df_clean.columns]
    df_clean = df_clean.drop_duplicates(subset=cols_check)

    # 3. Tratamento de Outliers via IQR
    if "sepal_width" in df_clean.columns:
        outlier_remover = IQROutlierRemover(target_col="sepal_width")
        df_clean = outlier_remover.fit_transform(df_clean)

    # 4. Imputação de Ausentes pela Mediana Agrupada
    num_cols = [c for c in ["sepal_length", "sepal_width", "petal_width"] if c in df_clean.columns]
    if num_cols and "species_clean" in df_clean.columns:
        imputer = GroupMedianImputer(group_col="species_clean", target_cols=num_cols)
        df_clean = imputer.fit_transform(df_clean)

    # 5. Engenharia de Features (Distorção de escala proposital)
    if "petal_length" in df_clean.columns and "petal_width" in df_clean.columns:
        df_clean["petal_area_mm2"] = df_clean["petal_length"] * df_clean["petal_width"] * 100.0

    return df_clean

def plot_diagnostic_graphics(df_raw: pd.DataFrame, df_clean: pd.DataFrame, X_train: pd.DataFrame, X_train_scaled: pd.DataFrame):
    """Exibe os gráficos com as sujeiras e o resultado pós-limpeza."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PAINEL DIAGNÓSTICO DE PRÉ-PROCESSAMENTO (UFU - BD II)", fontsize=16, fontweight='bold')

    # 1. Matriz de Ausência (Exibindo os nulos injetados)
    sns.heatmap(df_raw.isnull(), cbar=False, cmap="magma", ax=axes[0, 0], yticklabels=False)
    axes[0, 0].set_title("1. Mapeamento de Ausência de Dados (Dados Brutos)")

    # 2. Boxplot de Outliers
    col_outlier = "sepal_width" if "sepal_width" in df_raw.columns else df_raw.select_dtypes(include=np.number).columns[0]
    sns.boxplot(data=df_raw, y=col_outlier, ax=axes[0, 1], color="#60a5fa")
    axes[0, 1].set_title(f"2. Detecção de Outliers na variável '{col_outlier}'")

    # 3. Densidade SEM Padronização
    cols_plot = [c for c in ["sepal_length", "petal_area_mm2"] if c in X_train.columns]
    if len(cols_plot) > 1:
        sns.kdeplot(data=X_train[cols_plot], ax=axes[1, 0], fill=True)
        axes[1, 0].set_title("3. Densidade SEM Padronização (Escala Desproporcional)")
        axes[1, 0].set_xlabel("Valor Bruto")

        # 4. Densidade COM Padronização Z-Score
        sns.kdeplot(data=X_train_scaled[cols_plot], ax=axes[1, 1], fill=True)
        axes[1, 1].set_title("4. Densidade COM Padronização Z-score (Média=0, Std=1)")
        axes[1, 1].set_xlabel("Score Z")

    plt.tight_layout()
    plt.show()

    # Scatter Plot 3D Interativo
    if "sepal_length" in df_clean.columns and "petal_area_mm2" in df_clean.columns:
        print("\n[+] Gerando Gráfico 3D Interativo...")
        fig_3d = px.scatter_3d(
            df_clean, x='sepal_length', y='sepal_width', z='petal_area_mm2',
            color='species_clean', title="Scatter Plot 3D Interativo (Dados Sanitizados)"
        )
        fig_3d.write_html("iris_3d_interactive.html")
        print(" -> Salvo como 'iris_3d_interactive.html'.")

def main():
    print("=" * 80)
    print(" EXECUÇÃO DO PIPELINE DE BANCO DE DADOS II ")
    print("=" * 80)

    # 1. Carregamento dos dados
    df_raw = load_csv_data(CAMINHO_DO_SEU_CSV)

    # 2. Sanitização
    df_clean = run_data_cleaning_pipeline(df_raw)

    # 3. Separação para Aprendizado de Máquina
    features_num = [c for c in ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_area_mm2"] if c in df_clean.columns]
    target_col = "species_clean" if "species_clean" in df_clean.columns else df_clean.columns[-1]

    X = df_clean[features_num]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Escalonamento
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features_num)

    # 5. Exibição dos Gráficos Diagnósticos
    plot_diagnostic_graphics(df_raw, df_clean, X_train, X_train_scaled)

    # 6. Aprendizado de Máquina (KNN)
    knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_pred_raw = knn_raw.predict(X_test)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    pipe.fit(X_train, y_train)
    y_pred_pipe = pipe.predict(X_test)

    print("\n" + "=" * 80)
    print(" RESULTADOS DA CLASSIFICAÇÃO ML")
    print("=" * 80)
    print(f"Acurácia (Sem Padronização): {accuracy_score(y_test, y_pred_raw):.4f}")
    print(f"Acurácia (Com Padronização): {accuracy_score(y_test, y_pred_pipe):.4f}")
    print("\nRelatório Detalhado (Com Pipeline):")
    print(classification_report(y_test, y_pred_pipe))

    # 7. Matrizes de Confusão
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_raw), annot=True, fmt='d', ax=ax[0], cmap='Reds')
    ax[0].set_title("Matriz de Confusão (Dados Brutos)")
    ax[0].set_ylabel("Classe Real")
    ax[0].set_xlabel("Predição")

    sns.heatmap(confusion_matrix(y_test, y_pred_pipe), annot=True, fmt='d', ax=ax[1], cmap='Greens')
    ax[1].set_title("Matriz de Confusão (Dados Processados Z-Score)")
    ax[1].set_ylabel("Classe Real")
    ax[1].set_xlabel("Predição")
    plt.tight_layout()
    plt.show()

    # 8. Exportação
    df_clean.to_csv("iris_sanitizado_final.csv", index=False)
    print("\n[✔] PIPELINE EXECUTADO E FINALIZADO COM SUCESSO!")


if __name__ == "__main__":
    main()

"""# Nova seção"""

"""
===============================================================================
Aplicação no caso em que o usuário tem um dataset 
===============================================================================
"""

from typing import List
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from google.colab import files

RANDOM_STATE: int = 42
sns.set_theme(style="whitegrid", palette="deep")

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa valores ausentes utilizando a mediana agrupada por classe."""
    def __init__(self, group_col: str, target_cols: List[str]):
        self.group_col = group_col
        self.target_cols = target_cols
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.target_cols:
            self.medians_[col] = X.groupby(self.group_col)[col].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.target_cols:
            group_medians = X_out[self.group_col].map(self.medians_[col])
            X_out[col] = X_out[col].fillna(group_medians)
            X_out[col] = X_out[col].fillna(X_out[col].median())
        return X_out


class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """Substitui outliers por NaN usando a regra da Amplitude Interquartílica (IQR)."""
    def __init__(self, target_col: str, factor: float = 1.5):
        self.target_col = target_col
        self.factor = factor

    def fit(self, X: pd.DataFrame, y=None):
        q1 = X[self.target_col].quantile(0.25)
        q3 = X[self.target_col].quantile(0.75)
        iqr = q3 - q1
        self.lower_bound_ = q1 - self.factor * iqr
        self.upper_bound_ = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        outliers_mask = (X_out[self.target_col] < self.lower_bound_) | (X_out[self.target_col] > self.upper_bound_)
        X_out.loc[outliers_mask, self.target_col] = np.nan
        return X_out

def load_csv_data_colab() -> pd.DataFrame:
    """Solicita o envio do arquivo CSV do seu Mac diretamente no Colab."""
    print("=" * 80)
    print(" Por favor, clique no botão abaixo e selecione o arquivo 'iris_preprocessado.csv'")
    print("=" * 80)

    uploaded = files.upload()

    if not uploaded:
        raise FileNotFoundError("Nenhum arquivo foi selecionado!")

    nome_arquivo = list(uploaded.keys())[0]
    print(f"\n[+] Arquivo '{nome_arquivo}' carregado com sucesso no Google Colab!")

    df = pd.read_csv(io.BytesIO(uploaded[nome_arquivo]))
    return df

def run_data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executa o saneamento primário do dataset."""
    df_clean = df.copy()

   
    if "species_raw" in df_clean.columns:
        df_clean["species_clean"] = df_clean["species_raw"].astype(str).str.strip().str.lower()
    elif "species" in df_clean.columns:
        df_clean["species_clean"] = df_clean["species"].astype(str).str.strip().str.lower()

  
    cols_check = [c for c in ["sepal_length", "sepal_width", "petal_length", "petal_width", "species_clean"] if c in df_clean.columns]
    df_clean = df_clean.drop_duplicates(subset=cols_check)

    if "sepal_width" in df_clean.columns:
        outlier_remover = IQROutlierRemover(target_col="sepal_width")
        df_clean = outlier_remover.fit_transform(df_clean)

    num_cols = [c for c in ["sepal_length", "sepal_width", "petal_width"] if c in df_clean.columns]
    if num_cols and "species_clean" in df_clean.columns:
        imputer = GroupMedianImputer(group_col="species_clean", target_cols=num_cols)
        df_clean = imputer.fit_transform(df_clean)

    if "petal_length" in df_clean.columns and "petal_width" in df_clean.columns:
        df_clean["petal_area_mm2"] = df_clean["petal_length"] * df_clean["petal_width"] * 100.0

    return df_clean

def plot_diagnostic_graphics(df_raw: pd.DataFrame, df_clean: pd.DataFrame, X_train: pd.DataFrame, X_train_scaled: pd.DataFrame):
    """Exibe os gráficos diagnósticos."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PAINEL DIAGNÓSTICO DE PRÉ-PROCESSAMENTO (UFU - BD II)", fontsize=16, fontweight='bold')

    sns.heatmap(df_raw.isnull(), cbar=False, cmap="magma", ax=axes[0, 0], yticklabels=False)
    axes[0, 0].set_title("1. Mapeamento de Ausência de Dados (Dados Brutos)")

    col_outlier = "sepal_width" if "sepal_width" in df_raw.columns else df_raw.select_dtypes(include=np.number).columns[0]
    sns.boxplot(data=df_raw, y=col_outlier, ax=axes[0, 1], color="#60a5fa")
    axes[0, 1].set_title(f"2. Detecção de Outliers na variável '{col_outlier}'")

    cols_plot = [c for c in ["sepal_length", "petal_area_mm2"] if c in X_train.columns]
    if len(cols_plot) > 1:
        sns.kdeplot(data=X_train[cols_plot], ax=axes[1, 0], fill=True)
        axes[1, 0].set_title("3. Densidade SEM Padronização (Escala Desproporcional)")
        axes[1, 0].set_xlabel("Valor Bruto")

        sns.kdeplot(data=X_train_scaled[cols_plot], ax=axes[1, 1], fill=True)
        axes[1, 1].set_title("4. Densidade COM Padronização Z-score (Média=0, Std=1)")
        axes[1, 1].set_xlabel("Score Z")

    plt.tight_layout()
    plt.show()

    if "sepal_length" in df_clean.columns and "petal_area_mm2" in df_clean.columns:
        print("\n[+] Gerando Gráfico 3D Interativo...")
        fig_3d = px.scatter_3d(
            df_clean, x='sepal_length', y='sepal_width', z='petal_area_mm2',
            color='species_clean', title="Scatter Plot 3D Interativo (Dados Sanitizados)"
        )
        fig_3d.show()

def main():
    df_raw = load_csv_data_colab()

    df_clean = run_data_cleaning_pipeline(df_raw)

    features_num = [c for c in ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_area_mm2"] if c in df_clean.columns]
    target_col = "species_clean" if "species_clean" in df_clean.columns else df_clean.columns[-1]

    X = df_clean[features_num]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features_num)

    plot_diagnostic_graphics(df_raw, df_clean, X_train, X_train_scaled)

    knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_pred_raw = knn_raw.predict(X_test)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    pipe.fit(X_train, y_train)
    y_pred_pipe = pipe.predict(X_test)

    print("\n" + "=" * 80)
    print(" RESULTADOS DA CLASSIFICAÇÃO ML (DO SEU ARQUIVO CSV UPLOADED)")
    print("=" * 80)
    print(f"Acurácia (Sem Padronização): {accuracy_score(y_test, y_pred_raw):.4f}")
    print(f"Acurácia (Com Padronização): {accuracy_score(y_test, y_pred_pipe):.4f}")
    print("\nRelatório Detalhado (Com Pipeline):")
    print(classification_report(y_test, y_pred_pipe))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_raw), annot=True, fmt='d', ax=ax[0], cmap='Reds')
    ax[0].set_title("Matriz de Confusão (Dados Brutos)")
    ax[0].set_ylabel("Classe Real")
    ax[0].set_xlabel("Predição")

    sns.heatmap(confusion_matrix(y_test, y_pred_pipe), annot=True, fmt='d', ax=ax[1], cmap='Greens')
    ax[1].set_title("Matriz de Confusão (Dados Processados Z-Score)")
    ax[1].set_ylabel("Classe Real")
    ax[1].set_xlabel("Predição")
    plt.tight_layout()
    plt.show()

    df_clean.to_csv("iris_sanitizado_final.csv", index=False)
    print("\n[✔] PIPELINE CONCLUÍDO COM SUCESSO!")
    print("[+] Fazendo download automático do arquivo limpo 'iris_sanitizado_final.csv'...")
    files.download("iris_sanitizado_final.csv")


if __name__ == "__main__":
    main()