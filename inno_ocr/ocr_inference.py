import os
import time
import json
import glob
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from pdf2image import convert_from_path
from inno_ocr.craft import file_utils, imgproc
from inno_ocr.craft.test import test_net, copyStateDict
from inno_ocr.craft.craft import CRAFT
from inno_ocr.detectword import convertCoordination, read_img_by_coord
from inno_ocr.pre_dataset import RawDataset, AlignCollate
from inno_ocr.util import AttnLabelConverter
from inno_ocr.model import Model
from kiwipiepy import Kiwi

# 내부 글로벌 (lazy load)
oc_model = None
craft_model = None
converter = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_ocr(target_paths):
    global oc_model, craft_model, converter

    # 🔁 Lazy Load
    if craft_model is None or oc_model is None or converter is None:
        char_list = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑...(중략)...'

        converter = AttnLabelConverter(char_list)
        num_class = len(converter.character)

        craft_model = CRAFT()
        craft_model.load_state_dict(copyStateDict(torch.load('inno_ocr/craft/weights/craft_mlt_25k.pth', map_location=device)))
        craft_model = craft_model.to(device)
        if torch.cuda.is_available():
            craft_model = torch.nn.DataParallel(craft_model)
            torch.backends.cudnn.benchmark = False
        craft_model.eval()

        oc_model = Model(num_class)
        oc_model = torch.nn.DataParallel(oc_model).to(device)
        oc_model.load_state_dict(torch.load('./inno_ocr/weight/recognize/best_accuracy.pth', map_location=device))
        oc_model.eval()

    img_dir = './inno_ocr/img'
    os.makedirs(img_dir, exist_ok=True)
    data = []

    def image_scale(image):
        kernel_sharpening = np.array([[-1, -1, -1, -1, -1],
                                      [-1, 2, 2, 2, -1],
                                      [-1, 2, 4, 2, -1],
                                      [-1, 2, 2, 2, -1],
                                      [-1, -1, -1, -1, -1]]) / 8.0
        sharpended = cv2.filter2D(image, -1, kernel_sharpening)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(sharpended, kernel, iterations=1)
        _, thresh = cv2.threshold(cv2.bilateralFilter(erosion, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def detect():
        result_folder = './inno_ocr/result/'
        os.makedirs(result_folder, exist_ok=True)
        image_list, _, _ = file_utils.get_files(img_dir)
        for image_path in image_list:
            image = imgproc.loadImage(image_path)
            bboxes, polys, score_text = test_net(craft_model, image, 0.7, 0.4, 0.4,
                                                 torch.cuda.is_available(), False, None)
            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    # STEP 1: 전처리
    for path in target_paths:
        ext = os.path.splitext(path)[-1].lower()
        if ext == '.pdf':
            pages = convert_from_path(path, dpi=300, poppler_path='./inno_ocr/poppler-0.68.0/bin')
            for i, page in enumerate(pages):
                img_path = f'{img_dir}/{os.path.basename(path)[:-4]}_page{i+1:02d}.jpg'
                page.save(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                processed = image_scale(img)
                cv2.imwrite(img_path, processed)
        elif ext in ['.jpg', '.png']:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            processed = image_scale(img)
            img_name = os.path.basename(path)[:-4]
            cv2.imwrite(f'{img_dir}/{img_name}.jpg', processed)
        elif ext == '.tif':
            img = cv2.imread(path, 0)
            cv2.imwrite(f'{img_dir}/{os.path.basename(path)[:-4]}.jpg', img)

    # STEP 2: 텍스트 탐지
    detect()
    _, loc = convertCoordination()
    detect_output = read_img_by_coord(loc)
    img_list = glob.glob(f"{img_dir}/*.jpg")

    for _output, _target in zip(detect_output, img_list):
        filename = os.path.basename(_target)[:-4]
        sub_dir = f'./inno_ocr/temp/{filename}'
        os.makedirs(sub_dir, exist_ok=True)
        for j, img in enumerate(_output):
            _, encoded_img = cv2.imencode('.jpg', img)
            with open(f'{sub_dir}/{j}.jpg', mode='w+b') as f:
                encoded_img.tofile(f)

    AlignCollate_demo = AlignCollate(imgH=32, imgW=100)
    result = {}
    oc_model.eval()

    for i in img_list:
        filename = os.path.basename(i)[:-4]
        sub_data = []
        demo_data = RawDataset(root=f'./inno_ocr/temp/{filename}/')
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=128, shuffle=False, num_workers=0,
            collate_fn=AlignCollate_demo, pin_memory=True)

        for image_tensors, _ in demo_loader:
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([25] * image.size(0)).to(device)
            text_for_pred = torch.LongTensor(image.size(0), 26).fill_(0).to(device)

            preds = oc_model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS] if pred_EOS != -1 else pred
                pred_max_prob = pred_max_prob[:pred_EOS]
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item() if len(pred_max_prob) > 0 else 0
                    sub_data.append(pred)
                    data.append(pred)
                except:
                    continue

        result[filename] = " ".join(sub_data)

    with open('inno_ocr/result.txt', 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file)

    for folder in ['./inno_ocr/img', './inno_ocr/test']:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    for temp_folder in glob.glob('./inno_ocr/temp/*/'):
        shutil.rmtree(temp_folder)

    kiwi = Kiwi()
    return kiwi.space(" ".join(data), reset_whitespace=True)