
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(layout="wide")
st.title("📊 Развернутая аналитика оттока клиентов телеком-компании")

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    df = df.convert_dtypes().fillna(0)
    for col in df.columns:
        if "Int" in str(df[col].dtype):
            df[col] = df[col].astype("float64")
    return df

df = load_data()
st.subheader("📄 Предварительный просмотр данных")
st.dataframe(df.head())

st.write("### Последние строки")
st.dataframe(df.tail())

st.write("### Форма таблицы")
st.write(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

st.write("### Названия столбцов")
st.write(df.columns.tolist())

st.write("### Типы данных")
st.write(df.dtypes)

st.subheader("📊 Гистограммы признаков")
fig = plt.figure(figsize=(25, 25))
df.hist(ax=plt.gca(), bins=30)
st.pyplot(fig)

st.subheader("📈 Распределение целевого признака (class)")
fig, ax = plt.subplots()
df["class"].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

st.subheader("🔗 Корреляционная матрица")
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=1, ax=ax)
st.pyplot(fig)

st.subheader("📉 Распределение дневных расходов по оттоку")
fig, ax = plt.subplots()
sns.kdeplot(df.total_day_charge[df["class"] == 0], color="Red", fill=True, ax=ax)
sns.kdeplot(df.total_day_charge[df["class"] == 1], color="Blue", fill=True, ax=ax)
ax.legend(["Retain", "Churn"], loc="upper right")
ax.set_xlabel("Day Charges")
ax.set_ylabel("Density")
st.pyplot(fig)

X = df.drop(["class", "area_code", "phone_number"], axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("🎯 Важность признаков (Random Forest)")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
feat_scores = pd.DataFrame({"Importance": rf.feature_importances_}, index=X.columns)
feat_scores = feat_scores.sort_values(by="Importance")
fig, ax = plt.subplots(figsize=(10, 8))
feat_scores.plot(kind="barh", ax=ax)
st.pyplot(fig)

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

    st.subheader(f"📌 Модель: {name}")
    st.text(classification_report(y_test, y_pred))
    cm_fig, cm_ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=cm_ax)
    cm_ax.set_title("Матрица ошибок")
    st.pyplot(cm_fig)

st.subheader("📈 Сравнение моделей: ROC-кривые")
fig, ax = plt.subplots()
colors = ["orange", "red", "green", "blue", "purple"]
for (name, (fpr, tpr)), color in zip(roc_curves.items(), colors):
    ax.plot(fpr, tpr, linestyle="--", label=name, color=color)
ax.set_title("ROC-Кривая")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="best")
st.pyplot(fig)

st.subheader("📋 AUC по моделям")
for name, score in auc_scores.items():
    st.write(f"{name}: {round(score, 4)}")
