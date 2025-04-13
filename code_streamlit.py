import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(layout="wide")

# Чтение и очистка данных
@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    df = df.convert_dtypes().fillna(0)
    for col in df.columns:
        if "Int" in str(df[col].dtype):
            df[col] = df[col].astype("float64")
    return df

telecom_df = load_data()
st.title("Анализ оттока клиентов телеком-компании")
st.subheader("Предварительный просмотр данных")
st.dataframe(telecom_df.head())

# Гистограммы
st.subheader(" Распределения признаков")
fig = plt.figure(figsize=(20, 20))
telecom_df.hist(ax=plt.gca(), bins=30)
st.pyplot(fig)

# Корреляционная матрица
st.subheader("Корреляционная матрица")
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(telecom_df.corr(), annot=True, fmt=".2f", linewidths=1, ax=ax)
st.pyplot(fig)

# KDE plot
st.subheader("Распределение дневных расходов по оттоку")
fig, ax = plt.subplots()
sns.kdeplot(telecom_df.total_day_charge[telecom_df["class"] == 0], color="Red", fill=True, ax=ax)
sns.kdeplot(telecom_df.total_day_charge[telecom_df["class"] == 1], color="Blue", fill=True, ax=ax)
ax.legend(["Retain", "Churn"], loc="upper right")
ax.set_xlabel("Day Charges")
ax.set_ylabel("Density")
ax.set_title("Распределение дневных расходов")
st.pyplot(fig)

# Разделение признаков и целевой переменной
X = telecom_df.drop(["class", "area_code", "phone_number"], axis=1)
y = telecom_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "SVM": CalibratedClassifierCV(LinearSVC(max_iter=100000)),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

auc_scores = {}
roc_curves = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.calibrated_classifiers_[0].predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_scores[name] = auc
    roc_curves[name] = (fpr, tpr)
    st.subheader(f"Отчет по модели: {name}")
    st.text(classification_report(y_test, y_pred))
    cm_fig, cm_ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=cm_ax)
    st.pyplot(cm_fig)

# ROC-кривая
st.subheader("Сравнение моделей: ROC-кривая")
fig, ax = plt.subplots()
colors = ["orange", "red", "green", "blue", "purple"]
for (name, (fpr, tpr)), color in zip(roc_curves.items(), colors):
    ax.plot(fpr, tpr, linestyle="--", label=name, color=color)
ax.set_title("Receiver Operator Characteristics (ROC)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="best")
st.pyplot(fig)

st.subheader("AUC по моделям")
for name, score in auc_scores.items():
    st.write(f"{name}: {round(score, 4)}")
