### 2025년 5월 9일

# -*- coding: utf-8 -*-
"""
from fileinput import filename
from flask import Flask, render_template, request, jsonify
import glob
import json
import shutil
import os, time
from werkzeug.utils import secure_filename
import numpy as np
import ffmpeg
import base64
import uuid
from inno_stt.recognizer import Recognizer

from inno_mrc.model import main

from inno_ocr.craft.test import test_net,copyStateDict
from inno_ocr.craft.craft import CRAFT
from inno_ocr.ocr_inference import run_ocr_inference
from inno_ocr.util import AttnLabelConverter
from inno_ocr.pre_dataset import RawDataset, AlignCollate
from inno_ocr.model import Model
from pdf2image import convert_from_path
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
from kiwipiepy import Kiwi
"""

from flask import Flask, render_template, request, jsonify
import os, json, uuid, base64, glob, shutil, time
from werkzeug.utils import secure_filename
import numpy as np
import ffmpeg

# STT 전용
from inno_stt.stt_inference import run_stt

# NER 전용
# import tensorflow as tf
# from inno_ner import tokenization_kobert
# from transformers import TFBertModel
from inno_ner.ner_inference import run_ner

# MRC 전용
from inno_mrc.mrc_inference import run_mrc

# OCR 전용
from inno_ocr.ocr_inference import run_ocr

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("stt.html")

@app.route('/stt', methods=["GET"])
def stt():
    return render_template("stt.html")

@app.route('/inference_stt', methods=['POST'])
def inference_stt():
    data = request.get_json()
    text = run_stt(data["data"])
    return jsonify({"text": text})

@app.route('/ner', methods=["GET"])
def ner():
    return render_template("ner.html")

@app.route('/inference_ner', methods=["POST"])
def inference_ner():
    sentences = request.form.get("sentences")
    result = run_ner(sentences)
    return render_template("ner.html", files=result["tokens"], sen=result["sentence"], ans=result["entities"])

@app.route('/mrc', methods=["GET"])
def mrc():
    return render_template("mrc.html")

@app.route("/mrc_inference", methods=['GET', 'POST'])
def mrc_inference():
    if request.method == 'POST':
        query = request.form["query"]
        result = run_mrc(query)
    return render_template("mrc_result.html", query=query, result=result)

@app.route('/ocr', methods=["GET"])
def ocr():
    return render_template("ocr.html")

@app.route('/inference_ocr', methods=['POST'])
def inference_ocr():
    if 'file' not in request.files:
        return jsonify(result=render_template("result_file.html", filename="", result=""))

    file = request.files.get('file')
    os.makedirs('./inno_ocr/test/', exist_ok=True)
    filepath = f'./inno_ocr/test/{file.filename}'
    file.save(filepath)

    result = run_ocr([filepath])
    return jsonify(result=render_template("result_file.html", filename=file.filename, result=result))

@app.route('/buttons')
def buttons():
    return render_template("buttons.html")
@app.route('/cards')
def cards():
    return render_template("cards.html")

@app.route('/utilities-color')
def color():
    return render_template("utilities-color.html")
@app.route('/utilities-border')
def border():
    return render_template("utilities-border.html")
@app.route('/utilities-animation')
def animation():
    return render_template("utilities-animation.html")
@app.route('/utilities-other')
def other():
    return render_template("utilities-other.html")

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/register')
def register():
    return render_template("register.html")
@app.route('/forgot_password')
def forgot_password():
    return render_template("forgot-password.html")
@app.route('/not_found')
def not_found():
    return render_template("404.html")
@app.route('/blank')
def blank():
    return render_template("blank.html")

@app.route('/charts')
def charts():
    return render_template("charts.html")
@app.route('/tables')
def tables():
    return render_template("tables.html")


if __name__ == "__main__":
    app.run(debug=True)