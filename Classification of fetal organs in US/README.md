# Модель классификации органов плода на узи
___
##### сылка на датасет 💿
https://www.kaggle.com/datasets/vishwaskant786/2d-fetal-altrasound-images
##### сылка на модели 💿
https://drive.google.com/file/d/1KMVVvNpuDY6d1c-PbR-TPpze2SZa4YVJ/view?usp=sharing
##### dict обозначение данных на вход 🔽
- https://www.kaggle.com/datasets/vishwaskant786/2d-fetal-altrasound-images: путь до изображений органов плода на узи.
##### dict обозначение данных на выход 
- Class: _____; Probability: _______

##### образец данных на вход и выход 🔃
* вход: [матрица изображения cgray] | shape : (1646, 224, 224, 3)
* выход: [[0.20, 0.70, 0.04, 0.06]] | shape : (1646, 3)

##### метрики 📈

- accuracy: 0.95. модель: VGG16.

##### матрица ошибок 🔢
| TP | FN |
|------|------|
| FP | TN |

##### Значение фаилов 📄
* Classification of fetal organs in US : создание модели
* best_model_fetal.h5 : модель классификации


___
### Заметки ✏️
ваши коментарии
