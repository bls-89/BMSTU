
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
model_path = 'app'
# загрузка сохраненной модели Keras
model = keras.models.load_model(model_path)

# маршрут для отображения HTML страницы
@app.route('/')
def home():
    return render_template('index.html')

# маршрут для обработки POST запроса с введенными данными
@app.route('/predict', methods=['POST'])
def predict():
    # получение данных из формы
    input_data = request.form.to_dict()
    # преобразование данных в массив NumPy
    input_array = np.array(list(input_data.values())).astype(float)
    # изменение формы массива в форму, соответствующую форме входных данных модели
    input_array = input_array.reshape((1, 12))
    # прогнозирование значения переменной на основе введенных данных
    prediction = model.predict(input_array)[0][0]
    # форматирование вывода прогноза
    prediction_formatted = '{:.2f}'.format(prediction)
    # возврат прогноза в виде строки
    return prediction_formatted

if __name__ == '__main__':
    app.run(host='0.0.0.0')

#%%
