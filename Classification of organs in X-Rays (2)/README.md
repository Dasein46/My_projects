# Модель классификации частей тела на рентгене_2
___
##### сылка на датасет 💿
https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion
##### сылка на модели 💿
https://drive.google.com/file/d/1-CXF2aXdbmQuftz76mA1XeED2yp24w4l/view?usp=drive_link
##### dict обозначение данных на вход 🔽
- https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion/data: путь до изображений органов на рентгене.
##### dict обозначение данных на выход 
- Class: _____; Probability: _______

##### образец данных на вход и выход 🔃
* вход: [матрица изображения cgray] | shape : (1738, 224, 224, 3)
* выход: [[0.20, 0.70, 0.04, 0.06]] | shape : (1738, 3)

##### метрики 📈

- accuracy_multi : 0.990334. порог: 0.5. модель: нейронная модель resnet50.
- f1_macro: 0.796855. порог: 0.5. модель: нейронная модель resnet50.
- f1_samples: 0.884004. порог: 0.5. модель: нейронная модель resnet50.
- f1_micro: 0.899183. порог: 0.5. модель: нейронная модель resnet50.	
- f1_weighted: 0.889908. порог: 0.5. модель: нейронная модель resnet50.

##### матрица ошибок 🔢
| TP | FN |
|------|------|
| FP | TN |

##### Значение фаилов 📄
* Classification_of_organs_in_X-Rays_2 : создание модели
* my_model.pth : модель классификации


___
### Заметки ✏️
ваши коментарии
