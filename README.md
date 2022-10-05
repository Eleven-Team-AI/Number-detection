# Number-detection
Education project for master program Machine Learning Engineering by ITMO & AI Talent HUB.
Team: Voronkina Daria, Zhukov Dmitriy, Kuchuganova Svetlana
## Quick start examples
Clone repo and install [requirements.txt](https://github.com/Eleven-Team-AI/Number-detection/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Eleven-Team-AI/Number-detection  # clone
cd Number-detection
pip install -r requirements.txt  # install
```
Add video for detection and change path in [model.yaml](https://github.com/Eleven-Team-AI/Number-detection/blob/main/Enter_system/config/model.yaml)
```yaml
constants:
  path_video: path for your video
  frame_size: [2258,1244]
  frame_rate: 20
  coord: [[0,0], [0,1], [1,0], [1,1]] 
  path_enter_car: path for cars for enter system
```
Start module
```bash
python3 -m Enter_system
```
If car in list of cars for enter
```bash
MSK: 2022-10-05 16:12:06.988677+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:06.988677+03:00 - CAR detected - wait code detected
MSK: 2022-10-05 16:12:09.783503+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:09.783503+03:00 - CAR detected - wait code detected
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - code - 0841С
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - 0841С - passed
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - 0841С - passed
```
If car not in list cars for enter
```bash
MSK: 2022-10-05 16:12:06.988677+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:06.988677+03:00 - CAR detected - wait code detected
MSK: 2022-10-05 16:12:09.783503+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:09.783503+03:00 - CAR detected - wait code detected
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - start recording
MSK: 2022-10-05 16:12:10.441639+03:00 - CAR detected - code - 0841С
ALARM: CHECK CAR!!!!!!
```
## Models
Car detection  - [yolov5](https://github.com/ultralytics/yolov5). 

Plate detection - finetuned yolov5 on [Detecsi Plat NomorDataset](https://universe.roboflow.com/elektronika-instrumentasi-fisika-its/deteksi-plat-nomor/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

2 models were selected for OCR: [easyocr](https://github.com/jaidedai/easyocr) and [MORAN](https://github.com/Canjie-Luo/MORAN_v2). The final choice was made in favor of easyocr as showing the best results in conditions of low image quality.

Easyocr output example:

<img width="470" alt="Снимок экрана 2022-10-05 в 18 22 46" src="https://user-images.githubusercontent.com/55249362/194071378-6aeb6286-db26-4f15-aa70-2e0e86a08410.png">



MORAN output example:

<img width="470" alt="Снимок экрана 2022-10-05 в 18 23 50" src="https://user-images.githubusercontent.com/55249362/194071402-63d2aa2c-d4e5-4008-9d9f-d87bbade6880.png">


## Metrics
| model  |          task          | value |   metric |
|--------|:----------------------:|------:|---------:|
| yolov5 | number plate detection | 0.929 |  mAP 0.5 |
| yolov5 |     car detection      |  45.7 |  mAP 0.5 |
| easyocr|          OCR           |  93.  | Accuracy |
| MORAN  |          OCR           |  64.3  | Accuracy |


## Selection of hyperparameters for plat detection model

For experiments, we used Yolov5 nano.

| Epochs | learning rate |   opt | mAP 0.5 |
|--------|:-------------:|------:|--------:|
| 100    |     0.01      |   SGD |   0.922 |
| 100    |     0.001     |  Adam |   0.925 |
| 150    |     0.01      | AdamW |   0.929 |
We finetune this model with commad:
```bash
python train.py --img 640 --batch 32 --epochs 150 --data number_detection/data.yaml --weights yolov5n.pt --name num_detect_AdamW --optimizer AdamW 
```

