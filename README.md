# Computer Vision: классификация и сезонное преобразование изображений

Шесть исследовательских проектов на Python, TensorFlow/Keras и PyTorch/fastai.
Медицинские задачи относятся к классификации изображений; клиническая применимость не оценивалась.

| Проект | Модель | Протокол |
|---|---|---|
| [Области тела на рентгене](Classification%20of%20body%20parts%20in%20X-Rays/) | CNN, 5 классов | Манифест, разделение по пациентам |
| [Анатомические плоскости УЗИ плода](Classification%20of%20fetal%20organs%20in%20US/) | Замороженный VGG16 + dense, 4 класса | Patient ID из источника/манифеста |
| [Области тела на МРТ](Classification%20of%20organs%20in%20MRI/) | CNN, 4 класса | Манифест, разделение по пациентам |
| [Категории УЗИ](Classification%20of%20organs%20in%20US/) | CNN, 13 классов | Манифест; таксономия требует проверки |
| [Мультиразметочный рентген](Classification%20of%20organs%20in%20X-Rays%20(2)/) | ResNet50, 22 метки | Patient split, F1, фиксированный порог |
| [Лето ↔ зима на спутниковых снимках](Generation%20of%20satellite%20images/) | U-Net / ResNet34 | Разделение по городам до нарезки тайлов |

Архитектуры исходных экспериментов сохранены. Пересмотрены подготовка данных,
разбиения и оценка; старые outputs доступны в истории Git и не подтверждают новый код.
Полного повторного обучения на исходных данных пока нет.

## Запуск

Python 3.12. Клонируйте весь репозиторий: ноутбуки используют `cv_common`.
Создайте отдельное окружение и установите один набор зависимостей:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-keras.txt  # первые четыре классификатора
# или pip install -r requirements-torch.txt
pip install jupyterlab
jupyter lab
```

Откройте ноутбук нужного проекта. Для медицинских данных нужен CSV:

```csv
filename,label,patient_id,split
images/example_a.png,chest,sourceA_patient1,train
images/example_b.png,chest,sourceA_patient2,validation
images/example_c.png,chest,sourceA_patient3,test
```

Это схема, не готовый датасет. Пути относительны к DATA_ROOT. Метки зависят от проекта.
`split` необязателен: без него выполняется групповое разбиение с seed=12345.
`patient_id` обязателен; не заменяйте его именем снимка. Идентификаторы должны быть
обезличенными. Для мультиразметочного проекта метки берутся из train_df.csv.

В начале ноутбука можно задать переменные окружения через Python:

```python
import os
os.environ['DATA_ROOT'] = '/path/to/local/data'
os.environ['MANIFEST_PATH'] = '/path/to/local/manifest.csv'
os.environ['OUTPUT_DIR'] = '/path/to/local/artifacts/run01'
```

Для спутниковых изображений см. CLI в README проекта. Тесты:

```bash
python -m pytest tests/test_data.py tests/test_notebooks.py
# В окружении PyTorch:
python -m pytest tests/test_satellite.py
```

[Технический аудит](AUDIT.md) · [Проверки](VALIDATION.md).
Источники, ограничения и история метрик приведены в README каждого проекта.

## Дополнительная проверка исправлений

[Результат ревизии и требования к данным](REVIEW.md).
Каждый медицинский проект использует отдельную папку внутри artifacts/.
Для повторного обучения задайте новую пустую OUTPUT_DIR. Манифест разбиения
сохраняет относительные пути к DATA_ROOT и проверяемые контрольные суммы пикселей.
Для всех интеграционных тестов нужны оба набора зависимостей:

```bash
pip install -r requirements-keras.txt -r requirements-torch.txt
python -m pip check
python -m pytest -q
```
