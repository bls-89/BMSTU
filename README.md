# BMTSU
VKR BMTSU
Выпускная квалификационная работа Соколовского Б.Л. 
Application - папка с приложением flask и сохраненной моделью нейронной сети. Используется для размещения на web-сервере.
Датасет для ВКР_композиты - изначальные 2 датасета в формате .xlsx использованные для работы.
приложение - точная копия папки Application (изначально была в использовании. но на этапе деплоя приложения возникла ошибка из-за использования латиницы в названии.
code.ipynb - основной Jupyter notebook, использованный для работы с кодом.
df_apriori.csv - промежуточный датасет получшенный путем объединения по типу INNER двух датасетов из папки "Датасет для ВКР_композиты", очищенный от выбросов, но не нормализованный.
Weights-001--0.14651.hdf5 - файл с весами, используемый нейросетью для получения оптимального значения прогноза целевой переменной. Обращован в результате работы библиотеки checkpoint.
reqirements.txt - описание окружения и установленных библиотек.
