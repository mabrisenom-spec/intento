{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Análisis Exploratorio de Datos y Machine Learning Básico\n",
        "Notebook genérico para cargar datos, explorarlos y entrenar modelos básicos."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Carga de datos"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "ruta = 'datos.csv'  # Cambiar por su archivo\ndf = pd.read_csv(ruta)\ndf.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Exploración inicial"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(df.shape)\ndf.info()\ndf.describe(include='all').T"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "df.isnull().sum().sort_values(ascending=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "num_cols = df.select_dtypes(include=np.number).columns\nif len(num_cols) > 0:\n    df[num_cols].hist(figsize=(12,8))\n    plt.tight_layout()\n    plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "if len(num_cols) > 1:\n    plt.figure(figsize=(8,6))\n    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')\n    plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Machine Learning básico"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "target = 'objetivo'  # Cambiar por la variable objetivo\nX = df.drop(columns=[target])\ny = df[target]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report, accuracy_score\n\nnum_features = X.select_dtypes(include=np.number).columns\ncat_features = X.select_dtypes(exclude=np.number).columns\n\npreprocessor = ColumnTransformer([\n    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),\n    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)\n])\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = Pipeline([\n    ('prep', preprocessor),\n    ('model', RandomForestClassifier(random_state=42))\n])\n\nmodel.fit(X_train, y_train)\npred = model.predict(X_test)\n\nprint('Accuracy:', accuracy_score(y_test, pred))\nprint(classification_report(y_test, pred))"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}