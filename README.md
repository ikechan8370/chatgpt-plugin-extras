# chatgpt-plugin-extras

chatgpt-plugin一些花活额外工具

## 安装

`pip install -r requirements.txt`

`mkdir -p data/upload`

`python main.py`

运行在5000端口。目前默认使用CPU

## 使用

### ImageCaption

POST http://127.0.0.1:5000/image-captioning

Form-Data \
file: 图片文件

### Visual QA

POST http://127.0.0.1:5000/visual-qa

Form-Data \
file: 图片文件 \
q: 问题
