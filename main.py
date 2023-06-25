import os

import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
from flask import Flask, request, send_file
from uuid import uuid4
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

app = Flask('chatgpt-plugin-extras')


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=self.torch_dtype).to(self.device)

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class VQA:
    def __init__(self, device):
        print(f"Initializing Visual QA to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",
                                                              torch_dtype=self.torch_dtype).to(self.device)

    def inference(self, image_path, question):
        inputs = self.processor(Image.open(image_path), question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answers = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed Visual QA, Input Image: {image_path}, Output Text: {answers}")
        return answers


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs, output_filename):
        output_path = os.path.join('data', output_filename)
        image = Image.open(inputs)
        hed = self.detector(image)
        hed.save(output_path)
        print(f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {output_path}")
        return '/result/' + output_filename

class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    def inference(self, inputs, output_filename):
        output_path = os.path.join('data', output_filename)
        image = Image.open(inputs)
        hed = self.detector(image, scribble=True)
        hed.save(output_path)
        print(f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {output_path}")
        return '/result/' + output_filename

@app.route('/result/<filename>')
def get_result(filename):
    file_path = os.path.join('data', filename)
    return send_file(file_path, mimetype='image/png')


ic = ImageCaptioning("cpu")
vqa = VQA("cpu")
i2h = Image2Hed("cpu")
i2s = Image2Scribble("cpu")

@app.route('/image2hed', methods=['POST'])
def imag2hed():
    file = request.files['file']  # 获取上传的文件
    filename = str(uuid4()) + '.png'
    filepath = os.path.join('data', 'upload', filename)
    file.save(filepath)
    output_filename = str(uuid4()) + '.png'
    result = i2h.inference(filepath, output_filename)
    return result

@app.route('/image2Scribble', methods=['POST'])
def image2Scribble():
    file = request.files['file']  # 获取上传的文件
    filename = str(uuid4()) + '.png'
    filepath = os.path.join('data', 'upload', filename)
    file.save(filepath)
    output_filename = str(uuid4()) + '.png'
    result = i2s.inference(filepath, output_filename)
    return result


@app.route('/image-captioning', methods=['POST'])
def image_caption():
    file = request.files['file']  # 获取上传的文件
    filename = str(uuid4()) + '.png'
    filepath = os.path.join('data', 'upload', filename)
    file.save(filepath)
    result = ic.inference(filepath)
    return result


@app.route('/visual-qa', methods=['POST'])
def visual_qa():
    file = request.files['file']  # 获取上传的文件
    filename = str(uuid4()) + '.png'
    filepath = os.path.join('data', 'upload', filename)
    file.save(filepath)
    question = request.args.get('q')
    result = vqa.inference(filepath, question=question)
    return result





if __name__ == '__main__':
    app.run(host='0.0.0.0')
