# Мультиразметочная классификация рентгеновских снимков

[← Обзор](../README.md) · [Ноутбук](Classification_of_organs_in_X_Rays_2%20%281%29.ipynb)

## Задача и данные

22 анатомические метки; один снимок может принадлежать нескольким категориям.
Источник исходного проекта: [UNIFESP PNG dataset](https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion).
Лицензия и полнота выгрузки требуют проверки. DATA_ROOT содержит train_df.csv
(SOPInstanceUID, Target) и train/<SOPInstanceUID>-c.png.

Дополнительно нужен манифест `filename,patient_id[,split]`, где filename имеет вид
`train/<SOPInstanceUID>-c.png`. SOPInstanceUID — идентификатор изображения,
а не пациента. Если исходный набор не предоставляет проверенных групп пациентов,
новый независимый протокол нельзя выполнить без дополнительных метаданных.

## Метод

Python, PyTorch, fastai, pandas, scikit-learn. ResNet50 ImageNet, Resize(224),
обучающие аугментации fastai, MultiCategoryBlock и BCEWithLogitsLoss.
Learning rate finder использует train; лучшая эпоха выбирается по validation loss.
Групповое разделение примерно 60/20/20, seed=12345; официальный split сохраняется.
Проверяются пути, метки, пациентское пересечение и точные пиксельные дубли.

Порог 0,5 фиксирован. Инференс возвращает все метки выше порога, включая пустой
результат. На test вычисляются F1 macro/micro/samples, subset accuracy, Hamming loss
и precision/recall/F1/support по меткам. Elementwise accuracy не равна точности
целого набора меток и может быть высокой при большом числе отрицательных меток.

## Результаты

В старом README указаны F1 macro=0,796855, micro=0,899183, samples=0,884004,
weighted=0,889908, accuracy_multi=0,990334. Это исторические результаты при
случайном split по изображениям, не подтверждённый пациентский тест.
Новых результатов на реальных данных нет. Редкие метки, близкие дубли и смена
источника данных требуют отдельной оценки. Клиническая применимость не проверялась.

## Запуск

Установите `../requirements-torch.txt`. Задайте DATA_ROOT, MANIFEST_PATH, OUTPUT_DIR
по [общей инструкции](../README.md), запустите ноутбук сверху вниз.
Test оценивайте только после выбора гиперпараметров. Сохраняются split_manifest.csv,
model.pkl, classes.json и test_metrics.json. Загружайте лишь доверенные fastai/pickle-файлы.

[Аудит](../AUDIT.md) · [Проверки](../VALIDATION.md).
