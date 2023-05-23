import numpy as np
import rx
from rx import operators as ops

import whisper
import diart.operators as dops
from diart import OnlineSpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.sinks import DiarizationPredictionAccumulator


sample_rate = 16000
mic = MicrophoneAudioSource(sample_rate)

# Speech Recognition
asr_model = whisper.load_model('base')

step_asr = 0.5
duration_asr = 0.5

buffer_max_length = 10
audio_buffer = np.array([], dtype=np.float32)

speakers = [None, None]
new_speaker_flag = [False]

def update_audio_and_transcribe(audio_chunk):
    global audio_buffer
    audio_buffer = np.concatenate((audio_buffer, audio_chunk))
    result = asr_model.transcribe(audio_buffer)

    # Clear the current line
    print("\033[K", end="")

    # 如果 buffer 已經滿了，印出當前 speaker 的 text
    if len(audio_buffer) >= buffer_max_length * sample_rate:
        print(f"\r[{speakers[1]}] {result['text']}")
        audio_buffer = np.array([], dtype=np.float32)
    # 如果 speaker 已經改變，就印出前一位 speaker 的 text
    elif new_speaker_flag[0]:
        print(f"\r[{speakers[0]}] {result['text']}")
        audio_buffer = np.array([], dtype=np.float32)
        new_speaker_flag[0] = False
    else:
        print(f"\r[{speakers[1]}] {result['text']}", end="")


asr_stream = mic.stream.pipe(
    dops.rearrange_audio_stream(
        duration=duration_asr,
        step=step_asr,
        sample_rate=sample_rate),
    ops.map(lambda x: np.concatenate(x)),
    ops.map(update_audio_and_transcribe)
)

# Speaker Diarization
step_diarization = 3 # default: 3
duration_diarization = 5 # default: 5

diarization_buffer = []
diarization_buffer_max_length = 10

pipeline = OnlineSpeakerDiarization()
accumulator = DiarizationPredictionAccumulator(mic.uri)

# 當 speaker 改變時，印出語者名稱
def print_changed_speaker(annotation):
    get_speakers = [label for _, _, label in annotation.itertracks(yield_label=True)]
    speaker = get_speakers[-1]
    if speaker != speakers[1]:
        speakers[0] = speakers[1]
        speakers[1] = speaker
        new_speaker_flag[0] = True

def diarization(audio_chunk):
    global diarization_buffer
    diarization_buffer.append(audio_chunk)
    results = pipeline(np.concatenate(diarization_buffer))
    return results

diarization_stream = mic.stream.pipe(
    dops.rearrange_audio_stream(
        duration=duration_diarization,
        step=step_diarization,
        sample_rate=sample_rate),
    ops.buffer_with_count(count=1),
    ops.map(pipeline),
    ops.flat_map(lambda results: rx.from_iterable(results)),
    ops.do(accumulator)
)

# Subscribe
asr_stream.subscribe(
    on_next=lambda _: None,
    on_error=lambda e: print(e)
)

diarization_stream.subscribe(
    on_next=lambda _: print_changed_speaker(accumulator.get_prediction()),
    on_error=lambda e: print(e)
)

# Start microphone reading
mic.read()
