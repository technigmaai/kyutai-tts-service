import os
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("~/Development/vscode/modelscope/miopen_cache")
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base')
result = ans(
    'generated_speech-10.mp3',
    output_path='output_mp3_1.wav')
print("done")