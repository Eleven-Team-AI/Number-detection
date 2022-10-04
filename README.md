# Number-detection
Education project for master program Machine Learning Engineering by ITMO & AI Talent HUB.
Team: Voronkina Daria, Zhukov Dmitriy, Kuchuganova Svetlana
# Quick start examples
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
  path_enter_car: './Enter_system/src/enter_car.txt'
```
