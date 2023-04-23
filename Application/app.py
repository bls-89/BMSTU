from flask import Flask, render_template, request, jsonify

import keras

import numpy as np
app = Flask(__name__)

# Загрузка модели Keras
model_path = 'app'
model = keras.models.load_model(model_path)

# Создание объекта MinMaxScaler




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



    # Предсказание значения целевой переменной
    prediction = model.predict(input_array)

    # Обратное преобразование данных
    #prediction_unnormalized = scaler1.inverse_transform(prediction)

    # Возврат результата в формате JSON и ручной денормализатор, потому что задолбался уже с массивами воевать.
    return jsonify({'prediction': float((prediction*5.202339)+0.3894)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')