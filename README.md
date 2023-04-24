# BMSTU

Выпускная квалификационная работа Соколовского Б.Л. 

Application - папка с приложением flask и сохраненной моделью нейронной сети. Используется для размещения на web-сервере.ОСНОВНОЕ ПРИЛОЖЕНИЕ ЗДЕСЬ.

Another application - папка с приложением flask и впомогательными файлами для него. Используется для экспериментов c приложением.

non_deploy_app (render trouble)-LOCAL WORK - полностью ЛОКАЛЬНО работоспособная версия приложения со сложнейшей архитектурой нормализации входных данных и денормализаии предикта с помощью MinMaxScaler. Не работает на render.com ввиду отсуствия worker на послежднем.

Датасет для ВКР_композиты - изначальные 2 датасета в формате .xlsx, использованные для работы.

code.ipynb - основной Jupyter notebook, использованный для работы с кодом.

df_apriori.csv - промежуточный датасет получшенный путем объединения по типу INNER двух датасетов из папки "Датасет для ВКР_композиты", очищенный от выбросов, но не нормализованный.Используется как эталонный при номализации/денормализации в различных вариантах приложений flask.

Weights-001--0.14651.hdf5 - файл с весами, используемые нейросетью для получения оптимального значения прогноза целевой переменной. Образован в результате работы библиотеки checkpoint.

reqirements.txt - описание окружения и установленных библиотек.

___________________________________________________________________
ИНСТРУКЦИЯ ПО ЗАПУСКУ ПРИЛОЖЕНИЯ:

Приложение позволяет решать задачу прогнозирования "Соотношение матрица наполнитель". 

   •  скачать папку Application
   
   •	запустить app.py из скачанной папки
   
   •	в появившейся строке ( * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)) - нажать на ссылку: http://127.0.0.1:5000/. 
   
   •	В новом открывшемся окне (сайте) ввести 12 входных параметров и нажать "Прогноз".
   
   •	Насладиться чудесным музыкальным сопровождением при появлении прогнозируемого значения. 

