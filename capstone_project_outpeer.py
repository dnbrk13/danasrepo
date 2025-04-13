#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/dnbrk13/danasrepo/blob/main/capstone_project_outpeer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ЗАДАЧА ПРОГРАММИРОВАНИЯ #1: ИМПОРТ БИБЛИОТЕК/НАБОРОВ ДАННЫХ И ВЫПОЛНЕНИЕ ИССЛЕДОВАТЕЛЬСКОГО АНАЛИЗА ДАННЫХ

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import streamlit as st
from sklearn.preprocessing import StandardScaler


def show_histogram(df):
    st.header("📊 Гистограммы")
    fig = plt.figure(figsize=(30, 30))
    df.hist(figsize=(30, 30))
    st.pyplot(fig)

if __name__ == "__main__":
    st.title("📈 Анализ данных клиентов")

    df = pd.read_csv("telecom_churn.csv")
    df = df.convert_dtypes().fillna(0)

    st.subheader("📄 Предварительный просмотр данных")
    st.dataframe(df.head())

    # call plotting
    show_histogram(df)


# In[7]:


# чтение CSV file

telecom_df = pd.read_csv("telecom_churn.csv")

# Приводим все типы в совместимые форматы для Streamlit/Arrow
telecom_df = telecom_df.convert_dtypes()  # автоматическая оптимизация типов
telecom_df = telecom_df.fillna(0)         # заменяем NaN, чтобы Arrow не ругался

# Принудительно конвертируем все Int64Dtype в float64, чтобы убрать Arrow ошибку
for col in telecom_df.columns:
    if "Int" in str(telecom_df[col].dtype):
        telecom_df[col] = telecom_df[col].astype("float64")

# Fix serialization issues for Streamlit
telecom_df = telecom_df.convert_dtypes()  # Convert all columns to best supported types
telecom_df = telecom_df.fillna(0)         # Fill any NaNs to avoid Arrow errors



# In[8]:


# Загрузка 5 первых экземпляров
telecom_df.head()


# In[9]:


# Загрузка нижние 5 экземпляров
telecom_df.tail()


# In[10]:


# Проверка формы кадра данных
telecom_df.shape


# In[11]:


# Отображение название столбцов
telecom_df.columns


# In[12]:


# Получение сводки по типам данных
telecom_df.dtypes


# ВЫПОЛНЕНИЕ ВИЗУАЛИЗАЦИИ ДАННЫХ

# In[13]:


telecom_df.hist(figsize = (30, 30))
st.pyplot(plt.gcf())


# In[14]:


telecom_df["class"].value_counts()


# In[15]:


plt.figure(figsize = [10, 10])
telecom_df["class"].value_counts().plot(kind='pie')


# In[16]:


# Корреляционная матрица
corr_matrix = telecom_df.corr()
plt.figure(figsize = (15, 15))
cm = sns.heatmap(corr_matrix,
               linewidths = 1,
               annot = True,
               fmt = ".2f")
plt.title("Корреляционная матрица абонентов связи", fontsize = 20)
st.pyplot(plt.gcf())

# "voice_mail_plan" и "number_vmail_messages" сильно коррелируют.
#  «общий дневной заряд» и «общее количество ежедневных минут» сильно коррелируют.


# In[17]:


# Плата за отток по дням
ax = sns.kdeplot(telecom_df.total_day_charge[(telecom_df["class"] == 0)],
               color = "Red", fill=True)
ax = sns.kdeplot(telecom_df.total_day_charge[(telecom_df["class"] == 1)],
               color = "Blue", fill=True)

ax.legend(["Retain", "Churn"], loc = "upper right")
ax.set_ylabel("Density")
ax.set_xlabel("Day Charges")
ax.set_title("Распределение дневных расходов по оттоку")


# ОПРЕДЕЛЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ И ПОДГОТОВКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ МОДЕЛИ

# In[18]:


# Ненужные функции снизят скорость обучения, интерпретируемость модели и производительность обобщения на тестовых данных.
# Поэтому поиск и выбор наиболее полезных функций в наборе данных имеет решающее значение.
# Присвоение входных характеристик X и выходных (Churn) y

X = telecom_df.drop(["class", "area_code", "phone_number"], axis = "columns") # area_code and phone_number features are irrelevant to proceed further to train the model
y = telecom_df["class"]


# In[19]:


X.shape


# In[20]:


y.shape


# In[21]:


# Выполнение тренировочного сплита
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 150)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# In[24]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train.values.ravel())


# In[25]:


# Построение графика важности функции

feat_scores= pd.DataFrame({"Доля затронутых переменных" : rf.feature_importances_},index = X.columns)
feat_scores= feat_scores.sort_values(by = "Доля затронутых переменных")
feat_scores.plot(kind = "barh", figsize = (10, 5))
sns.despine()


# In[26]:


# Приведенный выше график сгенерирован алгоритмом Random Forest
# График показывает, что «total_day_minutes» возглавляет список важных функций, за ним следует «total_day_minutes» и так далее.


# ОБУЧЕНИЕ И ОЦЕНКА КЛАССИФИКАТОРА ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model_LR = LogisticRegression()

model_LR.fit(X_train, y_train)


# In[28]:


y_predict = model_LR.predict(X_test)


# In[29]:


print(classification_report(y_test, y_predict))
# точность - это отношение TP/(TP+FP)
# recall - это отношение TP/(TP+FN)
# F-beta score можно интерпретировать как взвешенное гармоническое среднее точности и полноты
# где оценка F-beta достигает своего лучшего значения в 1 и худшего в 0.


# ОБУЧЕНИЕ И ОЦЕНКА КЛАССИФИКАТОРА МЕТОДА ОПОРНЫХ ВЕКТОРОВ

# In[30]:


from sklearn.calibration import CalibratedClassifierCV # Для вывода оценки вероятности
from sklearn.svm import LinearSVC

model_svc = LinearSVC(max_iter=100000)
model_svm = CalibratedClassifierCV(model_svc)
model_svm.fit(X_train, y_train)


# In[31]:


y_predict = model_svm.predict(X_test)


# In[32]:


print(classification_report(y_test, y_predict))


# In[33]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# ОБУЧЕНИЕ И ОЦЕНКА СЛУЧАЙНОГО КЛАССИФИКАТОРА RandomForest

# In[34]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)


# In[35]:


y_predict = model_rf.predict(X_test)


# In[36]:


print(classification_report(y_test, y_predict))


# In[37]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# ОБУЧЕНИЕ И ОЦЕНКА KNeighborsClassifier

# In[38]:


from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)


# In[39]:

y_predict = model_knn.predict(X_test)


# In[40]:


print(classification_report(y_test, y_predict))


# In[41]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# ОБУЧЕНИЕ И ОЦЕНКА naive_bayes

# In[42]:


from sklearn.naive_bayes import GaussianNB


# In[43]:


model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)


# In[44]:


y_predict = model_gnb.predict(X_test)


# In[45]:


print(classification_report(y_test, y_predict))


# In[46]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# СРАВНЕНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ КЛАССИФИКАТОРОВ И ЗАКЛЮЧИТЕЛЬНЫЕ ЗАМЕЧАНИЯ

# In[47]:


model_LR.predict_proba(X_test)
  # Первый элемент - это вероятность того, что выход будет 0
  # Второй элемент - это вероятность того, что выход будет 1


# In[48]:


model_LR.predict_proba(X_test)[:, 1]


# In[49]:


y_test


# In[50]:


fpr1, tpr1, thresh1 = metrics.roc_curve(y_test, model_LR.predict_proba(X_test)[:, 1], pos_label= 1)


# In[51]:


fpr1


# In[52]:


tpr1


# In[53]:


thresh1


# In[54]:


# кривая ROC
from sklearn.metrics import roc_curve

fpr1, tpr1, thresh1 = roc_curve(y_test, model_LR.predict_proba(X_test)[:, 1], pos_label = 1)
fpr2, tpr2, thresh2 = roc_curve(y_test, model_svm.predict_proba(X_test)[:, 1], pos_label = 1)
fpr3, tpr3, thresh3 = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1], pos_label = 1)
fpr4, tpr4, thresh4 = roc_curve(y_test, model_knn.predict_proba(X_test)[:, 1], pos_label = 1)
fpr5, tpr5, thresh5 = roc_curve(y_test, model_gnb.predict_proba(X_test)[:, 1], pos_label = 1)


# In[56]:


# оценка AUC

from sklearn.metrics import roc_auc_score

auc_score1 = roc_auc_score(y_test, model_LR.predict_proba(X_test)[:, 1])
auc_score2 = roc_auc_score(y_test, model_svm.predict_proba(X_test)[:, 1])
auc_score3 = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
auc_score4 = roc_auc_score(y_test, model_knn.predict_proba(X_test)[:, 1])
auc_score5 = roc_auc_score(y_test, model_gnb.predict_proba(X_test)[:, 1])

print("Logistic Regression: ", auc_score1) # Логистическая регрессия
print("Support Vector Machine: ", auc_score2) # Метод опорных векторов
print("Random Forest: ", auc_score3) # Random Forest
print("K-Nearest Neighbors: ", auc_score4) # K-Nearest Neighbors
print("Naive Bayes: ", auc_score5) # Naive Bayes


# In[57]:


plt.plot(fpr1, tpr1, linestyle = "--", color = "orange", label = "Logistic Regression")
plt.plot(fpr2, tpr2, linestyle = "--", color = "red", label = "SVM")
plt.plot(fpr3, tpr3, linestyle = "--", color = "green", label = "Random Forest")
plt.plot(fpr4, tpr4, linestyle = "--", color = "yellow", label = "KNN")
plt.plot(fpr5, tpr5, linestyle = "--", color = "white", label = "Naive bayes")

plt.title('Receiver Operator Characteristics (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

plt.legend(loc = 'best')
plt.savefig('ROC', dpi = 300)
st.pyplot(plt.gcf())


# График показывает, что алгоритм Random Forest показал наилучшую AUC. Таким образом, очевидно, что модель Random Forest лучше справляется с классификацией отточенных/удержанных клиентов телекоммуникационных услуг.

# In[58]:


y_predict = model_rf.predict(X_test)
print(classification_report(y_test, y_predict))


# 

# In[59]:


if __name__ == "__main__":
    st.title("Capstone Data Analysis Dashboard")

    # Call your analysis code here, or modularize your code into functions
    st.subheader(" Preview Data")
    df = pd.read_csv("telecom_churn.csv")
    
    # Fix data types to avoid pyarrow serialization error
    df = df.convert_dtypes().fillna(0)
    st.dataframe(df.head())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use more iterations
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(max_iter=300)
model_LR.fit(X_train_scaled, y_train)

# Then predict with the scaled data
y_predict = model_LR.predict(X_test_scaled)