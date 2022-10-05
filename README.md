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
Add video for detection and change path in [model.yaml][https://github.com/Eleven-Team-AI/Number-detection/blob/main/Enter_system/config/model.yaml]
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
## Models
Car detection  - YoloV5.
Number plate detection - finetuned yolov5 on [ Detecsi Plat Nomor Dataset](https://universe.roboflow.com/elektronika-instrumentasi-fisika-its/deteksi-plat-nomor/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).
## Metrics
| model  |          task          | value |  metric |
|--------|:----------------------:|------:|--------:|
| yolov5 | number plate detection | 0.929 | mAP 0.5 |
| yolov5 |     car detection      |  45.7 | mAP 0.5 |
|     |          OCR           |  |         |
## Selection of hyperparameters for plat detection model
For experiments, we used Yolov5 nano.

| Epochs | learning rate |   opt | mAP 0.5 |
|--------|:-------------:|------:|--------:|
| 100    |     0.01      |   SGD |   0.922 |
| 100    |     0.001     |  Adam |   0.925 |
| 150    |     0.01      | AdamW |   0.929 |

