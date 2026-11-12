# Классификация областей тела на рентгеновских снимках

[← Обзор](../README.md) · [Ноутбук](Classification_of_body_parts_in_X_Rays.ipynb)

## Задача и данные

Классификация изображений 224×224 RGB. Классы: `head, chest, hand, shoulder, knee`.
Источники из исходного проекта (доступность, лицензии и полнота выгрузки не подтверждены):

- https://www.kaggle.com/datasets/dryari5/shoulder-xray-classification
- https://www.kaggle.com/datasets/antonbudnychuk/hand-xray
- https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

В старом README также перечислялись CT/MRI-наборы. Их связь с пятью рентгеновскими классами не установлена; чистоту модальности и источник каждого класса нужно проверить.

## Модель и метод

Два блока Conv2D/MaxPooling2D → Flatten → Dense(64) → softmax(5).
Python, TensorFlow/Keras, pandas, NumPy, PIL, scikit-learn, Matplotlib.
Adam, categorical crossentropy, выбор лучшей эпохи по validation loss.
Обучение учитывает все батчи. Validation/test не аугментируются.
Для VGG16 применяется штатная preprocess_input; для CNN — деление пикселей на 255.

Данные разделяются по patient_id; точные пиксельные дубли блокируют запуск.
Test отделён от выбора модели. Вычисляются precision/recall/F1/support по классам,
macro/weighted averages и матрица ошибок. Графики строятся только из нового history.

## Результаты и ограничения

Исторический запуск: 5 709 изображений, train 4 568 / validation 1 141; последняя val_accuracy 0,9894, val_loss 0,0464 за 3 эпохи. Это исходный split по снимкам, не независимый тест.

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
