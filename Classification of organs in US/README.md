# Классификация категорий УЗИ

[← Обзор](../README.md) · [Ноутбук](Classification_of_organs_in_US_.ipynb)

## Задача и данные

Классификация изображений 224×224 RGB. Классы: `benign, bladder, bowel, breast, gallbladder, heart, kidney, liver, malignant, salivary_gland, spleen, thyroid, venous_system`.
Источники из исходного проекта (доступность, лицензии и полнота выгрузки не подтверждены):

- https://www.kaggle.com/datasets/wangdaming1/classification-ultrasound-13-datastes

Таксономия смешивает органы с benign/malignant. Без проверки происхождения и меток модель нельзя описывать как классификатор только органов или как диагностическую систему.

## Модель и метод

Четыре Conv2D (32/64/128/256), три MaxPooling2D → Flatten → Dense(64) → softmax(13).
Python, TensorFlow/Keras, pandas, NumPy, PIL, scikit-learn, Matplotlib.
Adam, categorical crossentropy, выбор лучшей эпохи по validation loss.
Обучение учитывает все батчи. Validation/test не аугментируются.
Для VGG16 применяется штатная preprocess_input; для CNN — деление пикселей на 255.

Данные разделяются по patient_id; точные пиксельные дубли блокируют запуск.
Test отделён от выбора модели. Вычисляются precision/recall/F1/support по классам,
macro/weighted averages и матрица ошибок. Графики строятся только из нового history.

## Результаты и ограничения

Исторический запуск: train 2 256 / validation 560; последняя val_accuracy 0,8946 за 6 эпох. Использовались неполные батчи и split по изображениям.

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

Подробности исправлений — [AUDIT.md](../AUDIT.md), выполненные проверки — [VALIDATION.md](../VALIDATION.md).
