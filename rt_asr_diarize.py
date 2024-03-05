'''
將所有程式碼包裝成 class
加入speaker embedding
加入km incremental clustering

asr 流程說明
1. 從麥克風取得0.5秒音訊儲存在audio_buffer
2. 過濾掉靜音
3. 對整個audio_buffer進行語音辨識
4. 若辨識結果的segment數量超過上限，清除audio_buffer第一個segment的音訊
5. 檢查辨識出的最後一個segment的no_speech_prob，若超過閾值則不輸出(保留前一個時間點的輸出)
6. 將辨識結果覆蓋前一個時間點的segment，若buffer pop，則保留前一個時間點的第一個segment
'''

from rx import operators as ops
from diart.sources import MicrophoneAudioSource
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from km_clustering import IncrementalSpeakerClustering
import diart.operators as dops
import whisper
import torch
import numpy as np
import sys
import time

class OnlineDiarizeASR:
    def __init__(self, **kwargs):
        '''
        參數說明：

        [音訊處理相關參數]
        - step (float): 語音分段的步長（以秒為單位），預設值為 0.5。
        - duration (float): 語音分段的持續時間（以秒為單位），預設值為 0.5。
        - no_speech_prob_threshold (float): 語音段被視為無語音的概率閾值，預設值為 0.6。

        [Whisper相關參數]
        - whisper_model (str): 語音辨識模型的名稱，預設值為 'base'。
        - language (str): 語音辨識的語言，預設值為 None。
        - initial_prompt (str): 語音辨識的初始提示，預設值為 None。

        [Diarization相關參數]
        - n_speakers (int): 預期的語音輸入中的最大說話者數量，預設值為 20。
        - delta_new (float): 在語音分段中，兩個語音段之間的最大cosine距離，預設值為 0.8。
        '''

        # 音訊處理相關參數
        self.sample_rate = 16000
        self.step = kwargs.get('step', 0.5)
        self.duration = kwargs.get('duration', 0.5)
        self.no_speech_prob_threshold = kwargs.get('no_speech_prob_threshold', 0.6)
        
        # whisper 相關參數
        self.whisper_model = kwargs.get('whisper_model', 'base')
        self.language = kwargs.get('language', None)
        self.initial_prompt = kwargs.get('initial_prompt', None)

        # diairize 相關參數
        self.n_speakers = kwargs.get('n_speakers', 20)
        self.delta_new = kwargs.get('delta_new', 0.8)

        # 模型、麥克風
        self.mic = MicrophoneAudioSource(self.sample_rate)
        self.asr_model = whisper.load_model(self.whisper_model)
        self.emb_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cluster_model = IncrementalSpeakerClustering(
            delta_new=self.delta_new,
            max_speakers=self.n_speakers
        )
        
        # 內部狀態參數
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcription_results = []
        self.buffer_popped = False
        self.prev_segments_num = 1
        self.current_speaker = ''


    def update_audio_and_transcribe(self, audio_chunk: np.ndarray):
        # 將新的音訊chunk加入audio_buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))
        start_time = time.time()
        # 辨識整個audio_buffer
        result = self.asr_model.transcribe(
            self.audio_buffer,
            word_timestamps=True,
            initial_prompt=self.initial_prompt, 
            language=self.language
        )

        if len(result['segments']) > 2:
            # 按照segments時間點計算要pop掉的audio_buffer長度
            samples_to_pop = int(result['segments'][0]['end'] * self.sample_rate)
            reshaped_samples = torch.from_numpy(self.audio_buffer[:samples_to_pop][None,None])
            embedding = self.emb_model(reshaped_samples)
            embedding = embedding.squeeze()

            # 有做normalize --> 會變成0~1之間的值
            # l2_norm  = np.linalg.norm(embedding)
            # embedding = embedding / l2_norm

            self.current_speaker = self.cluster(embedding)
            
            # Pop the audio buffer
            self.audio_buffer = self.audio_buffer[samples_to_pop:]
            self.buffer_popped = True
        else:
            self.buffer_popped = False
        return result

    def is_speech(self, result: dict):
        if len(result['segments']) == 0:
            return False
        return result['segments'][-1]['no_speech_prob'] < self.no_speech_prob_threshold

    def print_result(self, result: dict):
        # 清除前一次的結果，從下面往上清除
        move_up_and_clear = '\r\033[K' + '\033[A\033[K'*(self.prev_segments_num-1)
        sys.stdout.write(move_up_and_clear)
        # sys.stdout.flush()

        output_list = [f"{seg['text']}" for seg in result['segments']]
        self.prev_segments_num = len(output_list)

        if self.buffer_popped:
            # print(f"Speaker {self.current_speaker}: {output_list[0]}")
            # 以顏色區分說話者
            print(f"\033[{self.current_speaker+31}m{output_list[0]}\033[0m")
            if len(output_list) > 1:
                joined_output = "\n".join(output_list[1:])
                self.prev_segments_num -= 1
                print(f'\r{joined_output}', end="", flush=True)
        else:
            # 覆蓋當前行
            joined_output = "\n".join(output_list)
            print(f'\r{joined_output}', end="", flush=True)

    def is_silence(self, audio_chunk: np.ndarray):
        return np.all(audio_chunk == 0)

    def cluster(self, embeddings: torch.Tensor):
        label = self.cluster_model.identify_single_speaker(embeddings)
        return label

    def __call__(self):
        # 訂閱麥克風的音訊流
        self.mic.stream.pipe(
            dops.rearrange_audio_stream(
                duration=self.duration,
                step=self.step,
                sample_rate=self.sample_rate
            ),
            ops.map(lambda x: np.concatenate(x)),
            ops.filter(lambda audio_data: not self.is_silence(audio_data)),
            ops.map(self.update_audio_and_transcribe),
            ops.filter(self.is_speech),
            ops.do_action(self.print_result)
        ).subscribe(
            on_next=lambda _: None,
            on_error=lambda e: print(e)
        )

        print("[開始]")
        self.mic.read()

if __name__ == '__main__':
    # params
    config = {
        'n_speakers': 20, # max_speakers
        'no_speech_prob_threshold': 0.5, 
        'delta_new': 0.7, # the maximum cosine distance between two speech segments
        'whisper_model': 'base',
        # 'initial_prompt': 'different speakers should be in different segments',
    }

    online_diarize_asr = OnlineDiarizeASR(**config)
    online_diarize_asr()
