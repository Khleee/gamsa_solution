### 2025년 5월 9일

# -*- coding: utf-8 -*-
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

from inno_ocr.craft.test import test_net,copyStateDict,str2bool
from inno_ocr.craft.craft import CRAFT
from inno_ocr.craft import file_utils, imgproc
from inno_ocr.detectword import convertCoordination, read_img_by_coord
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

## ocr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
craft = CRAFT()
print('Loading weights from checkpoint (inno_ocr/craft/weights/craft_mlt_25k.pth')
craft.load_state_dict(copyStateDict(torch.load('inno_ocr/craft/weights/craft_mlt_25k.pth', map_location=device)))
if device=='cuda':
            craft = craft.cuda()
            craft = torch.nn.DataParallel(craft)
            cudnn.benchmark = False
craft.eval()

#ocr text recognition
char_list = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝!@#$%^&*《》()[]【】【】\"\'◐◑oㅇ⊙○◎◉◀▶⇒◆■□△★※☎☏;:/.?<>-_=+×\￦|₩~,.㎡㎥ℓ㎖㎘→「」『』·ㆍ1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ읩①②③④⑤月日軍 '

converter = AttnLabelConverter(char_list)
num_class = len(converter.character)

ocr_model = Model(num_class)
ocr_model = torch.nn.DataParallel(ocr_model).to(device)
print('loading pretrained model from %s' % './inno_ocr/weight/recognize/best_accuracy.pth')
ocr_model.load_state_dict(torch.load('./inno_ocr/weight/recognize/best_accuracy.pth', map_location=device))

## stt
stt_recognizer = Recognizer(output_dir='./inno_stt/logs',
                        model_cfg='./inno_stt/configs/jasper10x5dr_sp_offline_specaugment.yaml',
                        ckpt='./inno_stt/results/Jasper_epoch60_checkpoint.pt',
                        task_path="./inno_stt/manifest",
                        vocab="./inno_stt/vocab",
                        decoding_mode='ctc_decoder',
                        )
stt_recognizer.load_model()

## ner
# 변수 + 토크나이저 + 모델 불러오기 + gpu 할당 설정
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from inno_ner import tokenization_kobert
from transformers import TFBertModel

tokenizer = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')
new_model = tf.keras.models.load_model("./inno_ner/kobert_tf2crf_all_es10",custom_objects={"TFBertModel":TFBertModel.from_pretrained("monologg/kobert", from_pt=True)})

# 라벨 사전
index_to_ner = {0: '-', 1: 'AC_B', 2: 'AC_I', 3: 'CT_B', 4: 'CT_I', 5: 'DR_B', 6: 'DR_I', 7: 'DT_B', 8: 'DT_I', 9: 'EV_B', 10: 'EV_I', 11: 'LC_B', 12: 'LC_I', 13: 'MY_B', 14: 'MY_I', 15: 'NOG_B', 16: 'NOG_I', 17: 'OG_B', 18: 'OG_I', 19: 'QT_B', 20: 'QT_I', 21: 'TI_B', 22: 'TI_I', 23: 'TX_B', 24: 'TX_I', 25: '[PAD]'}

# 라벨 -> 한국어화 사전
index_to_ner2 = {"LC":"지역","OG":"(민간)단체","NOG":"국가기관","DT":"날짜","DR":"기간","TI":"시간","AC":"법률","MY":"금액","QT":"수량","CT":"개수(빈도)","TX":"문서"}

# 문장길이
max_len = 178

## mrc

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("stt.html")

@app.route('/stt')
def stt():
    return render_template("stt.html")

@app.route('/inference_stt', methods=['POST'])
def inference_stt():
    data = request.get_json()
    dec_data = base64.b64decode(data["data"])
    os.makedirs('./inno_stt/logs', exist_ok=True)
    file_name = os.path.join('./inno_stt/logs', secure_filename(str(uuid.uuid4())))
    wav_file_name = file_name + '.wav'
    with open(file_name, mode='wb') as fd:
        fd.write(dec_data)
    _ = (ffmpeg.input(file_name).output(wav_file_name, format='wav', acodec='pcm_s16le', ac=1, ar=16000)
         .overwrite_output()
         .global_args('-hide_banner')
         .global_args('-loglevel', 'error')
         .run())

    text = stt_recognizer.transcribe(wav_file_name, option=1)

    os.remove(wav_file_name)
    os.remove(file_name)

    return jsonify({"text": text})

@app.route('/ner',methods=["GET","POST"])
def ner():
    if request.method == "POST":
        sentences = request.form.get("sentences")

        def ner_inference(test_sentence): # 문장 집어넣기
            result = []
            # 테스트 문장, 마스크 설정
            global tokenized_sentence, tokenized_mask
            
            tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, truncation=True, padding='max_length')])
            tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
            # 모델에 투입
            ans = new_model.predict([tokenized_sentence, tokenized_mask])
            tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
            
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, ans[0]):
                if (token.startswith("▁")): # _으로 시작하면
                    if token == "▁":
                        pass
                    else:
                        new_labels.append(index_to_ner[label_idx])
                        new_tokens.append(token[1:]) # ##뒤에 단어만 추가되게 설정
                # 문장시작,끝,패딩 은 넘어가기
                elif (token=='[CLS]'): 
                    pass
                elif (token=='[SEP]'):
                    pass
                elif (token=='[PAD]'):
                    pass
                # 앞, 뒤 토큰이 아니면 추가
                elif (token != '[CLS]' or token != '[SEP]'):
                    new_tokens.append(token)
                    new_labels.append(index_to_ner[label_idx])
            # 토큰, 라벨을 동시에 리스트에 저장
            for token, label in zip(new_tokens, new_labels):
                result.append([token, label])
            return result

        # 토큰,라벨 동시에 저장
        aa = ner_inference(sentences)
        
        # 개체명에 해당하는 단어들을 하나로 묶어 저장
        ans = []
        ans_text = ""
        for i,x in enumerate(aa):
            if x[1] != "-": # 개체명이 있다면
                if aa[i-1][1][:-2] == aa[i][1][:-2]: # 이전 개체명이랑 현재 개체명이랑 같다면
                    ans_text += x[0]
                elif aa[i-1][1][:-2] != aa[i][1][:-2]: #이전 개체명이랑 현재 개체명이 다르다면
                    ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
                    ans_text = ""
                    ans_text += x[0]
            elif x[1] == "-":
                ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
                ans_text = ""
        # 마지막 토큰까지 개체명인 경우 elif에 해당하지 않아서 ans에 추가되지 않으므로 개별로 설정
        if ans_text != "":
            ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
            ans_text = ""

        # " : " 요소를 제거하자
        ans2 = []
        for x in ans:
            if x != " : ":
                ans2.append(x)

        # 영어 개체명 -> 한국어 개체명
        for i,x in enumerate(ans2):
            if x[:2] in list(index_to_ner2.keys()):
                ans2[i] = ans2[i].replace(x[:2],index_to_ner2[x[:2]])
            elif x[:3] in list(index_to_ner2.keys()):
                ans2[i] = ans2[i].replace(x[:3],index_to_ner2[x[:3]])

        # 입력문장 불러오기
        bb = sentences

        # 테스트 출력
        print("입력 문장 :",bb)
        print("실제 태깅 형태 :",aa)

        ##

        return render_template("ner.html",
            files = aa,
            sen = bb,
            ans = ans2,
        )
    return render_template("ner.html")

@app.route('/mrc')
def mrc():
    return render_template("mrc.html")

@app.route("/mrc_inference", methods=['GET', 'POST'])
def mrc_inference():
    if request.method == 'POST':
        query = request.form["query"]
        result = main(query)
    return render_template("mrc_result.html", query=query, result=result)

@app.route('/ocr')
def ocr():
    return render_template("ocr.html")

@app.route('/inference_ocr', methods=['POST'])
def inference_ocr():
    def detect():
        result_folder = './inno_ocr/result/'
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        
        image_list, _, _ = file_utils.get_files('inno_ocr/img/')
        refine_net = None
        t = time.time()
        
        if device=='cuda':
            cuda = True
        else:
            cuda = False
                
        # load data
        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(craft, image, 0.7, 0.4, 0.4,
                                                cuda, False, refine_net)

            # save score text
            # filename, file_ext = os.path.splitext(os.path.basename(image_path))
            # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            # cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
        print("elapsed time : {}s".format(time.time() - t))
    
    def image_scale(image):
        kernel_sharpening = np.array([[-1, -1, -1, -1, -1],
                                    [-1, 2, 2, 2, -1],
                                    [-1, 2, 4, 2, -1],
                                    [-1, 2, 2, 2, -1],
                                    [-1, -1, -1, -1, -1]]) / 8.0
        sharpended = cv2.filter2D(image, -1, kernel_sharpening)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(sharpended, kernel, iterations=1)
        ret, thresh = cv2.threshold(cv2.bilateralFilter(erosion, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def run(target):
        data = []
        img_dir = './inno_ocr/img'
        for i in tqdm(target):
            ext = os.path.splitext(i)[-1]
            if ext.lower() == '.pdf':
                pages = convert_from_path(i, dpi=300, poppler_path='./inno_ocr/poppler-0.68.0/bin')
                for j, page in enumerate(pages):
                    page.save(f'{img_dir}/{os.path.basename(i)[:-4]}_page{j + 1:0>2d}.jpg')
                    img_path = f'{img_dir}/{os.path.basename(i)[:-4]}_page{j + 1:0>2d}.jpg'
                    img_array = np.fromfile(img_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    img = image_scale(img)
                    extension = '.jpg'
                    result, encoded_img = cv2.imencode(extension, img)
                    if result:
                        with open(f'{img_dir}/{os.path.basename(i)[:-4]}_page{j + 1:0>2d}.jpg', mode='w+b') as f:
                            encoded_img.tofile(f)
            elif ext.lower() == '.jpg' or ext.lower() == '.png':
                img_array = np.fromfile(i, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = image_scale(img)
                extension = '.jpg'
                result, encoded_img = cv2.imencode(extension, img)
                if result:
                    with open('{}/{}.jpg'.format(img_dir, os.path.basename(i)[:-4]), mode='w+b') as f:
                        encoded_img.tofile(f)
                # cv2.imwrite('{}/{}.jpg'.format(img_dir, os.path.basename(i)[:-4]), img)
            elif ext.lower() == '.tif':
                img = cv2.imread(i, 0)
                cv2.imwrite('{}/{}.jpg'.format(img_dir, os.path.basename(i)[:-4]), img)

        # 이미지 리스트
        img_list = glob.glob(img_dir + '/*.jpg')

        # text detect
        detect()
        
        len_loc, loc = convertCoordination()
        detect_output = read_img_by_coord(loc)

        for i, (_output, _target) in enumerate(zip(detect_output, img_list)):
            filename = os.path.basename(_target)[:-4]
            if os.path.exists("./inno_ocr/temp/{}".format(filename)) is False:
                os.makedirs("./inno_ocr/temp/{}".format(filename))
            for j, img in enumerate(_output):
                extension = '.jpg'
                # cv2.imwrite("./temp/{}/{}.jpg".format(filename, j), img)
                try:
                    result, encoded_img = cv2.imencode(extension, img)
                    if result:
                        with open('./inno_ocr/temp/{}/{}.jpg'.format(filename, j), mode='w+b') as f:
                            encoded_img.tofile(f)
                except:
                    print('오류난 파일 이름:', filename)

        AlignCollate_demo = AlignCollate(imgH=32, imgW=100)

        ocr_model.eval()
        result = {}
        with torch.no_grad():
            for i in img_list:
                sub_data = []
                filename = os.path.basename(i)[:-4]
                demo_data = RawDataset(root='./inno_ocr/temp/{}'.format(filename) + '/')  # use RawDataset
                demo_loader = torch.utils.data.DataLoader(
                    demo_data, batch_size=128,
                    shuffle=False,
                    num_workers=int(0),
                    collate_fn=AlignCollate_demo, pin_memory=True)
                with open(f'./inno_ocr/result/res_{filename}.txt', 'r', encoding='utf-8') as f:
                    bbox_text = f.readlines()

                for image_tensors, image_path_list in demo_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(device)
                    # For max length prediction
                    length_for_pred = torch.IntTensor([25] * batch_size).to(device)
                    text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(device)

                    preds = ocr_model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                    log = open(f'./inno_ocr/log_demo_result.txt', 'a', encoding='utf-8')
                    dashed_line = '-' * 80
                    head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    for idx, (img_name, pred, pred_max_prob) in enumerate(zip(image_path_list, preds_str, preds_max_prob)):
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        if '\\"' in pred:
                            pred = pred.replace('\\"','\"')
                        if "\\'" in pred:
                            pred = pred.replace("\\'",'\'')

                        pred_max_prob = pred_max_prob[:pred_EOS]

                        # calculate confidence score (= multiply of pred_max_prob)
                        try:
                            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                            sub_data.append(pred)
                            data.append(pred)
                            log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                        except:
                            print('오류나는 이미지 이름: ', img_name)
                    log.close()
                result[filename] = " ".join(sub_data)
        # print('result : ', result)
        with open('inno_ocr/result.txt', 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file)
        
        
        for file in os.listdir('./inno_ocr/img/'):
            os.remove("./inno_ocr/img/{}".format(file))
        for k in glob.glob('./inno_ocr/temp/*/'):
            shutil.rmtree(k)
        for file in os.listdir('./inno_ocr/test/'):
            os.remove("./inno_ocr/test/{}".format(file))
        
        
        kiwi = Kiwi()
        kiwi_result = kiwi.space(" ".join(data), reset_whitespace=True)
        # print('kiwi_result : ', kiwi_result)
        # print('result : ', " ".join(data))
        # return " ".join(data)
        return kiwi_result
        # return ""
    
    if 'file' not in request.files:
        print('No file part')
        return jsonify(result=render_template("result_file.html", filename = "", result=""))
    else:
        file = request.files.get('file')
        if not os.path.exists('./inno_ocr/test/'):
            os.makedirs('./inno_ocr/test/')
        file.save('./inno_ocr/test/' + file.filename)
        target = []
        target_dir = './inno_ocr/test/'
        target.append(target_dir + file.filename)
        result = run(target)
    return jsonify(result=render_template("result_file.html", filename = file.filename, result=result))

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