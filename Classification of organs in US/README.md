# Модель классификации органов на узи
___
##### сылка на датасет 💿
https://www.kaggle.com/datasets/wangdaming1/classification-ultrasound-13-datastes/data
##### сылка на модели 💿
https://drive.google.com/file/d/1sU1WO8pZHJ6890LkWcLZHfZmszkyYdlz/view?usp=sharing
##### dict обозначение данных на вход 🔽
- https://www.kaggle.com/datasets/wangdaming1/classification-ultrasound-13-datastes/data?select=data_13: путь до изображений органов на узи.
##### dict обозначение данных на выход 
- Class: _____; Probability: _______

##### образец данных на вход и выход 🔃
* вход: [матрица изображения cgray] | shape : (2816, 224, 224, 3)
* выход: [[0.20, 0.70, 0.04, 0.06]] | shape : (2816, 3)

##### метрики 📈

- accuracy: 0.90. модель: нейронная модель, построенная самостоятельно.

##### матрица ошибок 🔢
| TP | FN |
|------|------|
| FP | TN |

##### Значение фаилов 📄
* Classification_of_organs_in_US2 : создание модели
* best_model.h5 : модель классификации


___
### Заметки ✏️
ваши коментарии
