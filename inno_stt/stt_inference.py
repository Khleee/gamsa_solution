import os
import uuid
import base64
import ffmpeg
from inno_stt.recognizer import Recognizer

# Lazy-loading 글로벌 객체
stt_recognizer = None

def run_stt(base64_audio_bytes: str) -> str:
    """
    STT 추론을 수행하고 텍스트 반환
    :param base64_audio_bytes: base64로 인코딩된 오디오 데이터
    :return: 텍스트 추출 결과
    """
    global stt_recognizer

    # 모델이 로드되지 않았다면 최초 1회만 초기화
    if stt_recognizer is None:
        stt_recognizer = Recognizer(
            output_dir='./inno_stt/logs',
            model_cfg='./inno_stt/configs/jasper10x5dr_sp_offline_specaugment.yaml',
            ckpt='./inno_stt/results/Jasper_epoch60_checkpoint.pt',
            task_path="./inno_stt/manifest",
            vocab="./inno_stt/vocab",
            decoding_mode='ctc_decoder',
        )
        stt_recognizer.load_model()
        
    os.makedirs('./inno_stt/logs', exist_ok=True)

    # base64 -> binary -> wav 변환
    raw_data = base64.b64decode(base64_audio_bytes)
    temp_id = str(uuid.uuid4())
    input_file = f"./inno_stt/logs/{temp_id}.input"
    wav_file = f"./inno_stt/logs/{temp_id}.wav"

    with open(input_file, "wb") as f:
        f.write(raw_data)

    # ffmpeg: input.raw -> output.wav (mono, 16kHz, PCM)
    (
        ffmpeg
        .input(input_file)
        .output(wav_file, format='wav', acodec='pcm_s16le', ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    # STT 수행
    text = stt_recognizer.transcribe(wav_file, option=1)

    # 정리
    os.remove(input_file)
    os.remove(wav_file)

    return text
