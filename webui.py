import gradio as gr
from transformers import AutoTokenizer, AutoModel
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.asr.infer import ASRExecutor
import time
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()
model = model.eval()
#自动语音识别
asr = ASRExecutor()
#文字转语音
tts = TTSExecutor()
#对话记录
history_list = []
#语音输入
inputs_path = "./inputs"
#语音输出
outputs_path = "./outputs"

# 这个代码段将读入 input.wav 文件，然后转换文件的采样率为16000，单声道，采样深度为16位，并将其输出为 output.wav 文件。
def convert_wav(input_file, output_file):
    sound = AudioSegment.from_file(input_file)

    # set frame rate to 16000
    sound = sound.set_frame_rate(16000)

    # set channels to mono (1)
    sound = sound.set_channels(1)

    # set sample width to 2 (16 bit)
    sound = sound.set_sample_width(2)

    sound.export(output_file, format="wav")

# define functions
def chat_speech(input_text, input_audio):

    # 文本框输入不能为空
    txt = input_text
    # 没有内容输入
    if (len(str(txt).strip()) == 0) and (input_audio.__eq__(None)):
        txt = ""
        response = "你有什么想问的吗？"

    # 文本输入为空 并且 麦克风输入不为空，使用语音输入数据进行对话
    if len(str(txt).strip()) == 0 and input_audio.__ne__(None):
        rate, audio = input_audio
        is_exists = os.path.exists(inputs_path)
        if not is_exists:
            os.makedirs(inputs_path)
        # 将输入的语音保存为本地的wav文件
        input_audio_file_name = inputs_path + "/input_audio" + str(time.time()) + ".wav"
        sf.write(input_audio_file_name, audio, rate)
        # 将输入wav文件转换文件的采样率为16000，单声道，采样深度为16位的wav文件
        convert_audio_file_name = inputs_path + "/convert_audio" + str(time.time()) + ".wav"
        convert_wav(input_audio_file_name,convert_audio_file_name)
        # 将语音翻译为文本
        txt = asr(audio_file=convert_audio_file_name)

    # 文本输入不为空，使用文本输入数据进行对话
    if txt != "":
        response, history = model.chat(tokenizer, txt, history=history_list)
        history_list.append(history[len(history) - 1])

    # 创建输出文件夹
    is_exists = os.path.exists(outputs_path)
    if not is_exists:
        os.makedirs(outputs_path)
    # 语音输出文件
    audio_file_name = outputs_path + "/results" + str(hash(txt)) + str(time.time()) + ".wav"

    # use paddle_speech to generate speech data
    tts(text=response, am="fastspeech2_mix", lang="mix", output=audio_file_name)
    audio, sr = librosa.load(path=audio_file_name)
    # 拼接历史对话
    if len(history_list) != 0:
        separator = " "
        combined_history = separator.join([f"User：{x[0]} \n ChatSpeech:{x[1]} \n" for x in history_list])
    else:
        combined_history = ""
    # https://gradio.app/docs/#audio
    return (sr, audio,), response, combined_history


gradio_app = gr.Interface(
    fn=chat_speech,
    # 文本对话框 和 麦克风输入
    inputs=[gr.inputs.Textbox(lines=2, label="对话框", placeholder="有两种对话输入模式，一是文本输入，二是语音输入，如果两者都存在，只对文本输入进行响应。请问想聊点什么？"),
            gr.inputs.Audio(source="microphone")],
    outputs=[gr.components.Audio(label="语音回答"),
             gr.components.Textbox(label="文字回答"),
             gr.components.Textbox(label="对话历史",lines=10)],
    title="Chat Speech"
)


if __name__ == "__main__":
    gradio_app.launch()
