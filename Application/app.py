from flask import Flask, render_template, request, jsonify
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
app = Flask(__name__)

# Загрузка модели Keras
model_path = 'app'
model = keras.models.load_model(model_path)

# Создание объекта MinMaxScaler
scaler = MinMaxScaler()
scaler1 = MinMaxScaler()
df = pd.read_csv(r'static/df_apriori.csv')
y = df['Соотношение матрица-наполнитель']
X = df.drop(['Unnamed: 0','Соотношение матрица-наполнитель'], axis = 1)

# Определение маршрута для отображения HTML-страницы
@app.route('/')
def home():
    return render_template('index.html')

# Определение маршрута для обработки данных из формы HTML
@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы HTML
    input_data = request.form.to_dict()

    # Преобразование данных в массив numpy
    input_array = [[float(input_data['input{}'.format(i+1)]) for i in range(12)]]

    # Нормализация данных
    df_nr = scaler.fit(X)
    input_array_normalized = scaler.transform(input_array)

    # Предсказание значения целевой переменной
    prediction = model.predict(input_array_normalized)

    # Обратное преобразование данных
    #prediction_unnormalized = scaler1.inverse_transform(prediction)

    # Возврат результата в формате JSON и ручной денормализатор, потому что задолбался уже с массивами воевать.
    return jsonify({'prediction': float((prediction*5.202339)+0.3894)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')