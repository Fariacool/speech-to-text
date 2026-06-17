# FunASR Chinese Subtitle CLI

用 FunASR Paraformer 把中文长音频/视频转换成字幕。默认流水线是
`paraformer-zh + fsmn-vad + ct-punc`，可选说话人标注。

## 安装

一键安装 `uv`、`ffmpeg`、Python 环境和项目依赖：

```bash
./install.sh
```

如果要使用“两阶段说话人流程”（FunASR 分块 ASR + pyannote 全局 diarization + 合并），安装时加：

```bash
./install.sh --diarization
```

如果 GPU 镜像需要指定 PyTorch CUDA wheel 源：

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

CPU 测试环境：

```bash
./install.sh --cpu
```

安装脚本会在每一步输出时间戳和耗时，方便排查是卡在系统依赖、PyTorch wheel、项目依赖还是 diarization 依赖。

## 基本使用

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b
```

输出默认写到 `outputs/`：

- `*.srt`：给播放器或剪辑软件用
- `*.vtt`：给网页播放器用
- `*.txt`：快速阅读
- `*.segments.json`：标准化字幕片段
- `*.raw.json`：FunASR 原始输出，方便排查或二次处理

## 先本地抽音频再上传

可以直接把 MP3 传到 GPU 服务器上处理，不一定要上传很大的 MP4。脚本支持音频或视频输入；传入 MP3 后，脚本仍会先用 FFmpeg 转成 FunASR 更适合的 16 kHz 单声道 WAV 再识别。

你已经生成的高质量 MP3 可以直接使用：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -q:a 0 \
  -map a \
  example-6h.mp3
```

服务器上运行：

```bash
uv run --no-sync funasr-subtitle example-6h.mp3 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --hotword organization_b
```

不过 ASR 不需要音乐级音质。后续更推荐在本地直接导出体积更小的 16 kHz 单声道 MP3，上传更快，服务器预处理也更少：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a libmp3lame \
  -b:a 64k \
  example-6h-16k.mp3
```

如果上传体积不是问题，也可以直接生成 FunASR 最终会使用的 WAV，省掉服务器上的转码步骤：

```bash
ffmpeg -i "/path/to/source-video.mp4" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  example-6h-16k.wav
```

6 小时 WAV 通常会有几百 MB；综合看，`16k/64k mono MP3` 更适合临时上传到云 GPU。

## 6 小时访谈推荐命令

正式跑长任务前，建议先抽 5 分钟样本验证模型、字幕格式和说话人标签：

```bash
uv run --no-sync funasr-subtitle input.mp3 \
  --device cuda:0 \
  --spk \
  --preset-spk-num 2 \
  --sample-minutes 5 \
  --output-dir outputs-sample-spk \
  --prefix sample-spk
```

如果样本没问题，再跑完整音频。普通字幕可以不物理切分，让 FunASR 的 VAD 做内部长音频切分：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --hotword organization_b
```

需要说话人标签时加 `--spk`。不要同时加 `--chunk-minutes`，否则每个 chunk 会单独聚类，说话人编号会跨 chunk 错乱：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --spk \
  --preset-spk-num 2 \
  --hotword person_a \
  --hotword person_b
```

普通字幕长任务可以开启物理分块，这样日志能看到全局进度，且每个 chunk 完成后会更新 `*.partial.srt`、`*.partial.vtt`、`*.partial.txt`、`*.partial.segments.json`：

```bash
uv run --no-sync funasr-subtitle input.mp4 \
  --device cuda:0 \
  --chunk-minutes 45 \
  --batch-size-s 120 \
  --batch-threshold-s 30 \
  --max-single-segment-ms 30000 \
  --hotword person_a \
  --hotword person_b
```

后台运行可以用一行 `nohup`：

```bash
nohup bash -lc 'cd /path/to/speech-to-text && uv run --no-sync funasr-subtitle /path/to/input.mp3 --device cuda:0 --spk --preset-spk-num 2 --output-dir outputs-spk' > /path/to/speech-to-text/funasr-spk.log 2>&1 &
```

查看日志和中间结果：

```bash
tail -f /path/to/speech-to-text/funasr-spk.log
ls -lh /path/to/speech-to-text/outputs-spk/
```

## 说明

- 默认模型源是 Hugging Face：`--hub hf`，通常更适合海外服务器。
- 如果 ModelScope 更快，可以加 `--hub ms`。
- 使用 `--spk` 时，脚本会自动切到支持时间戳的 ModelScope Paraformer preset；说话人分离依赖句子时间戳。
- 已知说话人数时加 `--preset-spk-num`，两人访谈建议 `--preset-spk-num 2`。
- 长视频默认不需要先切分；脚本会先用 FFmpeg 抽取 16 kHz 单声道 WAV。
- 如果加了 `--chunk-minutes`，脚本会按块显示进度并持续写入 partial 输出；这个模式只适合不加 `--spk` 的普通字幕。
- 脚本默认禁止 `--spk + --chunk-minutes`。如果只是调试、接受 speaker 编号跨 chunk 不一致，可以显式加 `--allow-spk-chunking`。
- `--no-sync` 会让 uv 使用安装脚本准备好的 `.venv`，避免运行时重新解析依赖或覆盖手动安装的 PyTorch wheel。

## 更准确的两阶段说话人流程

`--spk` 不适合和物理分块一起使用。更稳的做法是：FunASR 分块做 ASR，pyannote 对整段音频做全局 diarization，然后按时间重叠合并 speaker 标签。

安装 diarization 依赖：

```bash
uv pip install --python .venv/bin/python -e '.[diarization]'
```

需要先在 Hugging Face 接受 pyannote 模型条款，并设置 token：

```bash
export HF_TOKEN=your_huggingface_token
```

推荐直接使用一键流程：

```bash
./run_two_stage.sh input.mp3 \
  --device cuda:0 \
  --diarization-device cuda \
  --chunk-minutes 30 \
  --num-speakers 5 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --output-dir outputs-two-stage
```

等价的 `uv` 命令：

```bash
uv run --no-sync two-stage-subtitle input.mp3 \
  --device cuda:0 \
  --diarization-device cuda \
  --segmentation-batch-size 32 \
  --embedding-batch-size 32 \
  --chunk-minutes 30 \
  --num-speakers 5 \
  --hotword person_a \
  --hotword person_b \
  --hotword organization_a \
  --output-dir outputs-two-stage
```

输出结构：

- `outputs-two-stage/asr/`：FunASR 分块 ASR 的原始输出和普通字幕
- `outputs-two-stage/diarization/`：pyannote 全局 speaker RTTM/JSON
- `outputs-two-stage/final/`：合并 speaker 后的最终字幕

后台运行：

```bash
nohup bash -lc 'cd /path/to/speech-to-text && ./run_two_stage.sh /path/to/input.mp3 --device cuda:0 --diarization-device cuda --num-speakers 5 --hotword person_a --output-dir outputs-two-stage' > logs/two-stage.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f logs/two-stage.log
```

所有 Python 脚本和 Bash 包装脚本都会输出类似这样的日志格式：

```text
[2026-06-17 12:00:00 +0000 elapsed=1h02m03s] [3/12] FunASR generate completed in 4m12s
```

日志会记录模型加载、音频抽取、ASR 推理、diarization 推理、speaker 合并、文件写入等步骤的耗时。

两阶段流程默认会给 FunASR 指定 timestamp-capable Paraformer 模型，并开启 `--require-timestamps`。如果 ASR 结果没有 `sentence_info` 句级时间戳，脚本会直接失败，避免生成每 30 分钟一条的巨长字幕。

pyannote 默认会用 `--segmentation-batch-size 32` 和 `--embedding-batch-size 32`，比上游默认的 `1` 更适合 24GB 级别 GPU。显存不够时把它们降到 `16` 或 `8`；如果想优先速度、接受轻微可复现性差异，可以加 `--allow-tf32`。

`community-1` 会返回 regular 和 exclusive 两种 diarization。脚本默认 `--diarization-output auto`，会优先使用更适合对齐字幕的 exclusive 输出；需要原始 speaker diarization 时可以改成 `--diarization-output regular`。

如果需要手动分步跑，命令如下。

第一步，分块跑 ASR，不加 `--spk`：

```bash
uv run --no-sync funasr-subtitle input.mp3 \
  --device cuda:0 \
  --model alextomcat/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --require-timestamps \
  --chunk-minutes 30 \
  --output-dir outputs-asr \
  --hotword person_a \
  --hotword person_b
```

第二步，对整段音频做全局 diarization，优先用 GPU：

```bash
uv run --no-sync pyannote-diarize input.mp3 \
  --device cuda \
  --segmentation-batch-size 32 \
  --embedding-batch-size 32 \
  --num-speakers 5 \
  --output-dir outputs-diarization
```

第三步，把全局 speaker 标签合并回 ASR 字幕：

```bash
uv run --no-sync merge-speakers \
  outputs-asr/input.segments.json \
  outputs-diarization/input.speakers.json \
  --output-dir outputs-final \
  --prefix input
```
