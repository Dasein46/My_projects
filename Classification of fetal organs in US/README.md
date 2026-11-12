# Классификация анатомических плоскостей УЗИ плода

[← Обзор](../README.md) · [Ноутбук](Classification_of_fetal_organs_in_US.ipynb)

## Задача и данные

Классификация изображений 224×224 RGB. Классы: `Fetal abdomen, Fetal brain, Fetal femur, Fetal thorax`.
Источники из исходного проекта (доступность, лицензии и полнота выгрузки не подтверждены):

- https://www.kaggle.com/datasets/vishwaskant786/2d-fetal-altrasound-images

По умолчанию читается DATA_ROOT/manifest.csv; MANIFEST_PATH задаёт другой путь.
Для исходного image_label.csv и images/ требуется ALLOW_FILENAME_PATIENT_IDS=1:
сначала подтвердите, что PatientNNNNN обозначает пациента, и проверьте официальный split.

## Модель и метод

VGG16 ImageNet без классификатора, замороженный encoder → Flatten → Dense(256) → softmax(4).
Python, TensorFlow/Keras, pandas, NumPy, PIL, scikit-learn, Matplotlib.
Adam, categorical crossentropy, выбор лучшей эпохи по validation loss.
Обучение учитывает все батчи. Validation/test не аугментируются.
Для VGG16 применяется штатная preprocess_input; для CNN — деление пикселей на 255.

Данные разделяются по patient_id; точные пиксельные дубли блокируют запуск.
Test отделён от выбора модели. Вычисляются precision/recall/F1/support по классам,
macro/weighted averages и матрица ошибок. Графики строятся только из нового history.

## Результаты и ограничения

В старом README была указана accuracy 0,95, но её нельзя считать проверенным независимым результатом. В сохранённом CSV было 1 646 изображений. Старый протокол и VGG-предобработка изменены.

**Новые метрики не получены:** необходимы исходные изображения и проверенные идентификаторы
пациентов. Изменённая предобработка и split требуют повторного обучения. Высокая accuracy
не доказывает переносимость на другие учреждения, аппараты и источники. Риск распознавания
источника вместо анатомии сохраняется при смешении датасетов. Близкие дубли не выявляются
точным хешированием. Клиническая применимость не проверялась.

## Воспроизведение

1. Установите `../requirements-keras.txt` по [общей инструкции](../README.md).
2. Подготовьте DATA_ROOT и CSV `filename,label,patient_id[,split]` с относительными путями.
3. Задайте MANIFEST_PATH и отдельный OUTPUT_DIR. Запустите ячейки сверху вниз.
4. Проверьте распределение классов по split; не подбирайте разбиение по test-результату.
5. Сохранённые best_model.keras, classes.json, history.json, split_manifest.csv и
   test_metrics.json остаются локально. Инференс использует то же преобразование RGB.

Методология — [METHODOLOGY.md](../METHODOLOGY.md), выполненные проверки — [VALIDATION.md](../VALIDATION.md).
