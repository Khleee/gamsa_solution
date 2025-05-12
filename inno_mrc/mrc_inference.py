# import argparse
import logging
from datasets import Dataset
import numpy as np
import torch
import re
from konlpy.tag import Okt
okt = Okt()
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
tfidf = TfidfVectorizer()


from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
)

# from tokenization_kobert import KoBertTokenizer

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
    # "kobert": (BertConfig, BertForQuestionAnswering, KoBertTokenizer)
    #"distilkobert": (DistilBertConfig, DistilBertForQuestionAnswering, KoBertTokenizer),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_mrc(input):
    question = input
    model_type = 'bert'
    model_name_or_path = './inno_mrc/KorQuAD_Gamsa_models'
    max_answer_length = 30
    max_seq_length = 384
    doc_stride = 128
    n_best_size = 20
    max_query_length = 64
    null_score_diff_threshold = 0.0
    config_name = ""
    cache_dir = ""
    tokenizer_name = ""

    # parser.add_argument(
    #     "--question",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Question to enter into the model.",
    # )
    # parser.add_argument(
    #     "--context",
    #     default=None,
    #     type=str,
    #     help="Context to enter into the model.",
    # )
    # parser.add_argument(
    #     "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    # )
    # parser.add_argument(
    #     "--tokenizer_name",
    #     default="",
    #     type=str,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    # )

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=False,
        cache_dir=cache_dir if cache_dir else None,
    )
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )
    model.to(DEVICE)
    
    def preprocess_validation_examples(examples):
        # questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            examples['question'],
            examples["context"],
            max_length=max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = inputs.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        inputs["example_id"] = []

        for i in range(len(inputs["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = inputs.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            inputs["example_id"].append([sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            inputs["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(inputs["offset_mapping"][i])
            ]
        return inputs
    
    # 명사 추출
    def extract_noun(text):
        """
        
        """
        # Read 'stopword.txt'
        txt = open("./inno_mrc/law_stopwords.txt", "r", encoding='UTF8') # 2022.01.04
        stopword = txt.read()
        txt.close()
        stopword_list = stopword.split('\n')

        noun = okt.nouns(text)
        # print(noun)

        # Stopword 및 한글자 형태소 제거 
        result = []
        for n in noun:
            if n not in stopword_list and 1 < len(n):
                result.append(n)

        # '[] 기호 제거
        result = re.sub(r"[\'\[\]\,]", '', str(result))
        return result
    
    import pickle
    
    with open('./inno_mrc/all_hang_2.pickle', 'rb') as f:
        all_law = pickle.load(f)
    
    hang_mtrx = tfidf.fit_transform(all_law['nouns'])
    features = tfidf.get_feature_names()
    
    input_noun = [t for t in extract_noun(question).split(' ') if t in features]
    input_noun = ' '.join(input_noun)
    
    input_mtrx = tfidf.transform([input_noun])

    cos_sim = cosine_similarity(hang_mtrx, input_mtrx)
    cos_flat = cos_sim.flatten()
    sim_rank_idx = cos_flat.argsort()[::-1]

    law = []
    jo = []
    hang = []
    hang_doc = []
    sim = []
    for rank, idx in enumerate(sim_rank_idx):
        # 상위 n위까지 보여주기?
        if rank == 20:
            break
        law.append(all_law.loc[idx, '법명'])
        jo.append(all_law.loc[idx, '조 이름'])
        hang.append(all_law.loc[idx, '항 번호'])
        hang_doc.append(all_law.loc[idx, '항 원본'])
        sim.append(cos_flat[idx])

    rank_df = pd.DataFrame((zip(law, jo, hang, hang_doc, sim)), columns = ['법명', '조 이름', '항 번호', '항 원본', '유사도'])

    # print('df', rank_df)
    # print(rank_df['조 원본'].values)
    valid_answers_list = []
    from torch.utils.data import DataLoader
    for i, df_context in enumerate(rank_df['항 원본'].values):
        print(f'i={i}')
        print('df_context', df_context)
        q = []
        c = []
        q.append(question)
        c.append(df_context)
        my_dict = {'question': q,
                'context': c}
        dataset = Dataset.from_dict(my_dict)
        dataset = dataset.map(preprocess_validation_examples, remove_columns=['question', 'context'], batched=True)
        dataset_offset_mapping = dataset['offset_mapping'][0]
    
        dataset = dataset.map(remove_columns=['offset_mapping', 'example_id'], batched=True)
    
        eval_dataloader = DataLoader(
        dataset, batch_size=1
        )
        for batch in tqdm(eval_dataloader):
            
            batch['input_ids'] = torch.tensor(batch['input_ids']).reshape(1, max_seq_length).to(DEVICE)
            batch['token_type_ids'] = torch.tensor(batch['token_type_ids']).reshape(1, max_seq_length).to(DEVICE)
            batch['attention_mask'] = torch.tensor(batch['attention_mask']).reshape(1, max_seq_length).to(DEVICE)
            
            # print('batch',batch['input_ids'].shape)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'], attention_mask = batch['attention_mask'])
            
    # start_logits, end_logits = outputs.start_logits.argmax(dim=-1), outputs.end_logits.argmax(dim=-1)
    # print('start_logits', start_logits)
    # print('end_logits', end_logits)
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        context = df_context
    # print(dataset['input_ids'][0][start_logits:end_logits+1])
    # decoded = tokenizer.decode(dataset['input_ids'][0][start_logits:end_logits+1])
    
    # print('decoded', decoded)
        n_best_size = n_best_size
    #Gather the indices the best start/end logits:
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(dataset_offset_mapping)
                    or end_index >= len(dataset_offset_mapping)
                    or dataset_offset_mapping[start_index] is None
                    or dataset_offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                    start_char = dataset_offset_mapping[start_index][0]
                    end_char = dataset_offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "source": rank_df.loc[i, '법명']+' '+rank_df.loc[i, '조 이름'][:rank_df.loc[i, '조 이름'].find('(')]+' '+str(rank_df.loc[i, '항 번호'])+'항',
                            "context": context,
                            "answer": context[start_char:end_char]
                        }
                    )
        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:1]
        # print('valid_answers', valid_answers)
        valid_answers_list.append(valid_answers)
        print('valid_answers_list', valid_answers_list)
    return valid_answers_list
    
if __name__ == "__main__":
    main()