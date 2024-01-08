# Модель классификации частей тела на снимках
___
##### сылка на датасет 💿
https://www.kaggle.com/datasets/chenghanpu/brain-tumor-mri-and-ct-scan
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
https://www.kaggle.com/code/polomarco/chest-ct-segmentation-lung-heart-trachea
https://www.kaggle.com/datasets/dryari5/shoulder-xray-classification
https://www.kaggle.com/datasets/antonbudnychuk/hand-xray
https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
##### сылка на модели 💿
https://drive.google.com/file/d/1CRiFZTfKanN6YyVsimc2zvOQNzUWCoIR/view?usp=drive_link
##### dict обозначение данных на вход 🔽
- https://drive.google.com/drive/folders/1VJjYoUftSS_p2OlujqIhNgldi5Bpsv7k?usp=drive_link: путь до изображения снимка плеча x-ray;
- https://drive.google.com/drive/folders/13JMCGEmC98prbBTb2lsa9M_k-E6j5bP4?usp=drive_link: путь до изображения снимка колена x-ray;
- https://drive.google.com/drive/folders/1ZZvih-UTAk9p3OYVxbnhDitJiK24jVJZ?usp=drive_link: путь до изображения снимка головы x-ray;
- https://drive.google.com/drive/folders/1Ruy3-7buJl6L6sbsio-IHLInRS_zlg-Y?usp=drive_link: путь до изображения снимка руки x-ray;
- https://drive.google.com/drive/folders/1_nm-WVvAazAIPLKe3gRa8R3BG_XMOnuz?usp=drive_link: путь до изображения снимка грудной клетки x-ray.

##### dict обозначение данных на выход 🔼
- Filename: Predictions

##### образец данных на вход и выход 🔃
* вход: [матрица изображения RGB] | shape : (5709, 128, 128, 3)
* выход: [[0.20, 0.70, 0.02, 0.05, 0.03]] | shape : (5709, 3)

##### метрики 📈

- acc : 0.9894. порог: 0.6455790400505066. модель: нейронная модель с 7 слоями.

##### матрица ошибок 🔢
| TP | FN |
|------|------|
| FP | TN |

##### Значение фаилов 📄
* classification_of_body_parts_in_X_Rays.ipynb : создание модели
* my_model.h5 : модель классификации


___
### Заметки ✏️
ваши коментарии
