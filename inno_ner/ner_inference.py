import numpy as np
import tensorflow as tf
from inno_ner import tokenization_kobert
from transformers import TFBertModel

# 🔁 Lazy 로딩용 글로벌 변수
ner_model = None
ner_tokenizer = None

# 라벨 사전
index_to_ner = {
    0: '-', 1: 'AC_B', 2: 'AC_I', 3: 'CT_B', 4: 'CT_I', 5: 'DR_B', 6: 'DR_I',
    7: 'DT_B', 8: 'DT_I', 9: 'EV_B', 10: 'EV_I', 11: 'LC_B', 12: 'LC_I',
    13: 'MY_B', 14: 'MY_I', 15: 'NOG_B', 16: 'NOG_I', 17: 'OG_B', 18: 'OG_I',
    19: 'QT_B', 20: 'QT_I', 21: 'TI_B', 22: 'TI_I', 23: 'TX_B', 24: 'TX_I', 25: '[PAD]'
}

index_to_ner2 = {
    "LC": "지역", "OG": "(민간)단체", "NOG": "국가기관", "DT": "날짜", "DR": "기간",
    "TI": "시간", "AC": "법률", "MY": "금액", "QT": "수량", "CT": "개수(빈도)", "TX": "문서"
}

max_len = 178

def run_ner(sentences: str):
    """NER 추론 수행 및 결과 반환"""
    global ner_model, ner_tokenizer

    if ner_model is None or ner_tokenizer is None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        ner_tokenizer = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')
        ner_model = tf.keras.models.load_model(
            "./inno_ner/kobert_tf2crf_all_es10",
            custom_objects={"TFBertModel": TFBertModel.from_pretrained("monologg/kobert", from_pt=True)}
        )

    tokenized_sentence = np.array([
        ner_tokenizer.encode(sentences, max_length=max_len, truncation=True, padding='max_length')
    ])
    tokenized_mask = np.array([[int(x != 1) for x in tokenized_sentence[0].tolist()]])

    ans = ner_model.predict([tokenized_sentence, tokenized_mask])
    tokens = ner_tokenizer.convert_ids_to_tokens(tokenized_sentence[0])

    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, ans[0]):
        if token in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        if token.startswith("▁"):
            token = token[1:] if token != "▁" else ""
        new_tokens.append(token)
        new_labels.append(index_to_ner[label_idx])

    # 개체명 병합
    result = []
    ans = []
    ans_text = ""
    for i, (token, label) in enumerate(zip(new_tokens, new_labels)):
        if label != "-":
            if i > 0 and new_labels[i-1][:-2] == label[:-2]:
                ans_text += token
            else:
                if ans_text:
                    ans.append(f"{new_labels[i-1][:-2]} : {ans_text}")
                ans_text = token
        else:
            if i > 0 and new_labels[i-1] != "-":
                ans.append(f"{new_labels[i-1][:-2]} : {ans_text}")
                ans_text = ""
    if ans_text:
        ans.append(f"{new_labels[-1][:-2]} : {ans_text}")

    # label -> 한글
    ans2 = []
    for x in ans:
        label_key = x.split(' : ')[0]
        label_kor = index_to_ner2.get(label_key, label_key)
        ans2.append(x.replace(label_key, label_kor))

    return {
        "tokens": list(zip(new_tokens, new_labels)),
        "sentence": sentences,
        "entities": ans2
    }