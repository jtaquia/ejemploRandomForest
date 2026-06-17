import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, 
                             r2_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración de la página
st.set_page_config(page_title="Auto-ML & ROC Curve App", layout="wide")

st.title("🤖 Plataforma de Entrenamiento y Evaluación de Modelos")
st.write("Sube tu dataset, selecciona el algoritmo, ajusta hiperparámetros y analiza las métricas junto a la Curva ROC.")

# ----------------------------------------------------
# 1. Barra lateral para la carga de datos
# ----------------------------------------------------
st.sidebar.header("📁 1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv","xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("👀 Vista previa del Dataset")
    st.dataframe(df.head())
    
    # Selección de Tipo de Tarea
    st.sidebar.header("🎯 2. Configuración de la Tarea")
    task_type = st.sidebar.selectbox("Selecciona el tipo de problema", ["Clasificación", "Regresión"])
    
    # Selección de variables
    all_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox("Selecciona la variable objetivo (Y)", all_columns)
    feature_columns = [col for col in all_columns if col != target_column]
    selected_features = st.sidebar.multiselect("Selecciona las variables predictoras (X)", feature_columns, default=feature_columns)
    
    if not selected_features:
        st.warning("⚠️ Por favor, selecciona al menos una variable predictora.")
    else:
        # Preprocesamiento
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        # Codificación de variables categóricas
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                
        if y.dtype == 'object' and task_type == "Clasificación":
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
        
        # Relleno de nulos básico
        X = X.fillna(X.median(numeric_only=True))
        
        # Escalamiento (muy importante para Regresión Logística)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # División de datos
        test_size = st.sidebar.slider("Tamaño del set de validación (%)", 10, 50, 20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # ----------------------------------------------------
        # 2. Selección de Modelo e Hiperparámetros
        # ----------------------------------------------------
        st.sidebar.header("🧠 3. Modelo e Hiperparámetros")
        
        if task_type == "Clasificación":
            model_choice = st.sidebar.selectbox("Selecciona el algoritmo", ["Random Forest", "Árbol de Decisión", "Regresión Logística"])
            
            # Hiperparámetros dinámicos según el modelo
            if model_choice == "Random Forest":
                n_estimators = st.sidebar.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
                max_depth = st.sidebar.slider("Máxima Profundidad (max_depth)", 1, 50, 10)
                criterion = st.sidebar.selectbox("Criterio (criterion)", ["gini", "entropy", "log_loss"])
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42, n_jobs=-1)
                
            elif model_choice == "Árbol de Decisión":
                max_depth = st.sidebar.slider("Máxima Profundidad (max_depth)", 1, 50, 10)
                criterion = st.sidebar.selectbox("Criterio (criterion)", ["gini", "entropy", "log_loss"])
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
                
            elif model_choice == "Regresión Logística":
                penalty = st.sidebar.selectbox("Penalización (penalty)", ["l2", "none"])
                C_val = st.sidebar.number_input("Inversa de fuerza de regularización (C)", value=1.0, min_value=0.01, step=0.1)
                model = LogisticRegression(penalty=penalty if penalty != "none" else None, C=C_val, max_iter=1000, random_state=42)
                
        else: # Regresión
            model_choice = st.sidebar.selectbox("Selecciona el algoritmo", ["Random Forest", "Árbol de Decisión", "Regresión Lineal"])
            
            if model_choice == "Random Forest":
                n_estimators = st.sidebar.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
                max_depth = st.sidebar.slider("Máxima Profundidad (max_depth)", 1, 50, 10)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                
            elif model_choice == "Árbol de Decisión":
                max_depth = st.sidebar.slider("Máxima Profundidad (max_depth)", 1, 50, 10)
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                
            elif model_choice == "Regresión Lineal":
                st.sidebar.text("La Regresión Lineal no posee hiperparámetros críticos.")
                model = LinearRegression()

        # ----------------------------------------------------
        # 3. Entrenamiento y Visualización de Resultados
        # ----------------------------------------------------
        if st.button("🚀 Entrenar Modelo"):
            st.subheader(f"📊 Resultados - {model_choice}")
            
            with st.spinner("Entrenando el modelo..."):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            
            # --- CASO CLASIFICACIÓN ---
            if task_type == "Clasificación":
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Exactitud)", f"{acc:.4f}")
                col2.metric("F1-Score (Weighted)", f"{f1:.4f}")
                
                # Sección de Gráficos (Importancia y Curva ROC)
                st.write("---")
                g_col1, g_col2 = st.columns(2)
                
                # Importancia de variables (Solo para árboles)
                with g_col1:
                    if model_choice in ["Random Forest", "Árbol de Decisión"]:
                        st.subheader("📈 Importancia de Variables")
                        importances = model.feature_importances_
                        feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
                        st.bar_chart(feat_importances)
                    elif model_choice == "Regresión Logística":
                        st.subheader("📈 Coeficientes del Modelo")
                        coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=True)
                        st.bar_chart(coefs)
                
                # Curva ROC
                with g_col2:
                    st.subheader("📈 Curva ROC")
                    n_classes = len(np.unique(y))
                    
                    # Verificar si es clasificación binaria
                    if n_classes == 2:
                        # Obtener probabilidades de la clase positiva
                        y_probs = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_probs)
                        roc_auc = auc(fpr, tpr)
                        
                        # Dibujar con Matplotlib
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
                        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
                        ax.set_title('Receiver Operating Characteristic (ROC)')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                    else:
                        st.info("ℹ️ La curva ROC está configurada en este código para clasificación binaria (2 clases). Para datasets multiclase, se requiere un análisis One-vs-Rest.")

            # --- CASO REGRESIÓN ---
            else:
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                
                col1, col2 = st.columns(2)
                col1.metric("RMSE (Error Cuadrático Medio Raíz)", f"{rmse:.4f}")
                col2.metric("R² Score (Coef. de Determinación)", f"{r2:.4f}")
                
                # Gráficos para regresión
                st.write("---")
                if model_choice in ["Random Forest", "Árbol de Decisión"]:
                    st.subheader("📈 Importancia de Variables")
                    importances = model.feature_importances_
                    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
                    st.bar_chart(feat_importances)
else:
    st.info("💡 Por favor, sube un archivo CSV desde la barra lateral para comenzar.")