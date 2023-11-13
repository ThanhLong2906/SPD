from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import moviepy.editor

# video_source = "./video/khach_moi_for_sound.mp4"#"../ghostfacenet-colab/video/thuy_minh.mp4"
# # sound extractor
# sound = "./sound/speaker_2.wav"
# audio = moviepy.editor.VideoFileClip(video_source).audio
# audio.write_audiofile(sound)

# initialize pipeline
inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
    reversion="v1.0.5",
    sv_model="damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch",
    sv_model_revision="v1.2.2",
)

# input: a list of audio in which the first item is a speech recording to detect speakers, 
# and the following wav file are used to extract speaker embeddings.
audio_list = [
    # "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/record.wav",
    # "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk1.wav",
    # "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk2.wav",
    # "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk3.wav",
    # "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/speaker_diarization/spk4.wav",
    "./sound/video_sound.wav",
    "./sound/speaker_1.wav",
    "./sound/speaker_2.wav",
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)