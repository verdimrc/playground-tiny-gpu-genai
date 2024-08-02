# Playground: dev/test models on RTX A1000 6GB

This document provides super succint notes on trying-out various ML/AI/GenAI algos/models on a
single GPU with tiny memory.

## 1. FAISS

Pre-requisite: download the sift1M dataset
([ref](https://github.com/facebookresearch/faiss/blob/main/benchs/README.md#getting-sift1m)).

```console
% /usr/bin/time gunzip bigann_base.bvecs.gz
770.15user 64.59system 15:43.28elapsed 88%CPU (0avgtext+0avgdata 1780maxresident)k
191298912inputs+257812504outputs (0major+236minor)pagefaults 0swaps

% /usr/bin/time python3 bench_gpu_sift1m.py
load data
Loading sift1M...done
============ Exact search
add vectors to index
warmup
benchmark
k=1 0.133 ms, R@1 0.9914
k=2 0.127 ms, R@1 0.9930
k=4 0.121 ms, R@1 0.9932
k=8 0.120 ms, R@1 0.9932
k=16 0.120 ms, R@1 0.9932
k=32 0.121 ms, R@1 0.9922
k=64 0.124 ms, R@1 0.9931
k=128 0.130 ms, R@1 0.9932
k=256 0.152 ms, R@1 0.9931
k=512 0.166 ms, R@1 0.9923
k=1024 0.230 ms, R@1 0.9931
============ Approximate search
train
WARNING clustering 100000 points to 4096 centroids: please provide at least 159744 training points
add vectors to index
warmup
benchmark
nprobe=   1 0.004 ms recalls= 0.3818 0.4169 0.4169
nprobe=   2 0.005 ms recalls= 0.4981 0.5538 0.5538
nprobe=   4 0.008 ms recalls= 0.6051 0.6872 0.6872
nprobe=   8 0.013 ms recalls= 0.6960 0.8068 0.8068
nprobe=  16 0.022 ms recalls= 0.7606 0.8954 0.8954
nprobe=  32 0.040 ms recalls= 0.7998 0.9548 0.9548
nprobe=  64 0.071 ms recalls= 0.8159 0.9826 0.9826
nprobe= 128 0.133 ms recalls= 0.8239 0.9956 0.9956
nprobe= 256 0.243 ms recalls= 0.8270 0.9996 0.9996
nprobe= 512 0.482 ms recalls= 0.8273 1.0000 1.0000
375.32user 7.13system 0:53.81elapsed 710%CPU (0avgtext+0avgdata 1336632maxresident)k
1453568inputs+8outputs (1620major+185185minor)pagefaults 0swaps

# NOTE: next file is based on bench_gpu_sift1m.py, but with the GPU usage disabled.
% /usr/bin/time python3 bench_cpu_sift1m.py
load data
Loading sift1M...done
============ Exact search
add vectors to index
warmup
benchmark
k=1 2.672 ms, R@1 0.9914
k=2 1.750 ms, R@1 0.9914
k=4 1.000 ms, R@1 0.9914
k=8 0.982 ms, R@1 0.9914
k=16 0.998 ms, R@1 0.9914
k=32 1.233 ms, R@1 0.9914
k=64 1.202 ms, R@1 0.9914
k=128 1.383 ms, R@1 0.9914
k=256 1.160 ms, R@1 0.9914
k=512 1.431 ms, R@1 0.9914
k=1024 1.495 ms, R@1 0.9914
============ Approximate search
train
WARNING clustering 100000 points to 4096 centroids: please provide at least 159744 training points
add vectors to index
warmup
benchmark
nprobe=   1 0.007 ms recalls= 0.3824 0.4173 0.4173
nprobe=   2 0.009 ms recalls= 0.4980 0.5537 0.5537
nprobe=   4 0.014 ms recalls= 0.6059 0.6875 0.6875
nprobe=   8 0.027 ms recalls= 0.6956 0.8063 0.8063
nprobe=  16 0.047 ms recalls= 0.7602 0.8950 0.8950
nprobe=  32 0.078 ms recalls= 0.7973 0.9546 0.9546
nprobe=  64 0.143 ms recalls= 0.8142 0.9825 0.9825
nprobe= 128 0.254 ms recalls= 0.8223 0.9956 0.9956
nprobe= 256 0.480 ms recalls= 0.8248 0.9996 0.9996
nprobe= 512 0.915 ms recalls= 0.8251 1.0000 1.0000
4205.51user 332.46system 3:52.14elapsed 1954%CPU (0avgtext+0avgdata 1497772maxresident)k
9728inputs+0outputs (119major+76972minor)pagefaults 0swaps
```

## 2. Nemo: MCore

First, visit <https://huggingface.co/meta-llama/Llama-2-7b-hf> to download the tokenizers files
(i.e., `tokenizer.json` and `tokenizer.model`). Registration required.

Next, download a subset of the
[Law-StackExchange](https://huggingface.co/datasets/ymoslem/Law-StackExchange) dataset. This dataset
is under the CC BY-SA 4.0 license, so you can use it for any purpose, including commercial use,
without attribution.

```bash
mkdir -p llama2-tokenizer
# Place `tokenizer.json` and `tokenizer.model` to the current directory.

# Download sample dataset
wget https://huggingface.co/datasets/scooterman/hf-law-qa-dataset/resolve/main/law-qa-curated.zip
unzip law-qa-curated.zip
```

Preprocess the `.jsonl` files:

```console
$ mkdir -p law-qa-curated-v3-07212024-withtags-nosynth/{train,val,test}/

$ ./prep-data.sh
...
Opening /data/law-qa-train.jsonl
Time to startup: 0.07660889625549316
Processed 1000 documents (5317.976475301856 docs/s, 11.865617348989417 MB/s).
Processed 2000 documents (6286.125365689895 docs/s, 13.97702748069637 MB/s).
Processed 3000 documents (6417.737827226332 docs/s, 14.150858441554467 MB/s).
Processed 4000 documents (6710.089241398122 docs/s, 14.779634579411965 MB/s).
Processed 5000 documents (6713.634156187444 docs/s, 14.760035227217795 MB/s).
Processed 6000 documents (6827.709418511996 docs/s, 15.034956781865287 MB/s).
Processed 7000 documents (6936.686239408664 docs/s, 15.300349408057059 MB/s).
Processed 8000 documents (6972.335018524637 docs/s, 15.335808548969249 MB/s).
Processed 9000 documents (7004.240165717521 docs/s, 15.40939902163724 MB/s).
Processed 10000 documents (7016.378392592955 docs/s, 15.403686526786437 MB/s).
Processed 11000 documents (7009.961433876411 docs/s, 15.422945772380713 MB/s).
Processed 12000 documents (7073.708212133465 docs/s, 15.547896768560989 MB/s).
0.02user 0.00system 0:15.02elapsed 0%CPU (0avgtext+0avgdata 30644maxresident)k
45936inputs+0outputs (248major+1415minor)pagefaults 0swaps

...
Opening /data/law-qa-val.jsonl
Time to startup: 0.07482767105102539
Processed 1000 documents (4752.143918115868 docs/s, 12.415538860179943 MB/s).
Processed 2000 documents (5574.826048526978 docs/s, 14.340752161517349 MB/s).
0.01user 0.00system 0:12.97elapsed 0%CPU (0avgtext+0avgdata 31152maxresident)k
144inputs+0outputs (0major+1585minor)pagefaults 0swaps

...
Opening /data/law-qa-test.jsonl
Time to startup: 0.07889676094055176
Processed 1000 documents (5129.938002453484 docs/s, 13.241129972248483 MB/s).
Processed 2000 documents (6045.33383491266 docs/s, 15.392186748937206 MB/s).
0.02user 0.00system 0:12.66elapsed 0%CPU (0avgtext+0avgdata 30900maxresident)k
184inputs+0outputs (1major+1549minor)pagefaults 0swaps
```

Because we use docker (instead of [enroot](https://github.com/NVIDIA/enroot)), we need to change the
group ownership of the `.{bin,idx}` files from `root` to a regular user.

```bash
sudo chown $(id -u):$(id -g) law-qa-curated-v3-07212024-withtags-nosynth/{train,val,test}/*

find law-qa-curated-v3-07212024-withtags-nosynth/ -iname '*.idx' -o -iname '*.bin' | xargs ls -al
# Should not see files in root group anymore.
```

Now, let's launch a tiny-scale pretraining.

```bash
$ SEQ_LENGTH=128 MAX_POSITION_EMBEDDINGS=128 GLOBAL_BATCH_SIZE=32 ./pretrain.sh
...
training ...
[before the start of training step] datetime: 2024-08-01 09:44:51
 [2024-08-01 09:44:55] iteration        1/       5 | consumed samples:           32 | elapsed time per iteration (ms): 3902.7 | throughput per GPU (TFLOP/s/GPU): 0.2 | learning rate: 6.000000E-05 | global batch size:    32 | loss scale: 4294967296.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
Number of parameters in transformer layers in billions:  0.01
Number of parameters in embedding layers in billions: 0.03
Total number of parameters in billions: 0.05
Number of parameters in most loaded shard in billions: 0.0452
Theoretical memory footprints: weight and optimizer=776.41 MB
[Rank 0] (after 1 iterations) memory (MB) | allocated: 542.3623046875 | max allocated: 623.57861328125 | reserved: 648.0 | max reserved: 648.0
 [2024-08-01 09:44:56] iteration        2/       5 | consumed samples:           64 | elapsed time per iteration (ms): 689.4 | throughput per GPU (TFLOP/s/GPU): 1.0 | learning rate: 6.000000E-05 | global batch size:    32 | loss scale: 2147483648.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
 [2024-08-01 09:44:56] iteration        3/       5 | consumed samples:           96 | elapsed time per iteration (ms): 645.4 | throughput per GPU (TFLOP/s/GPU): 1.1 | learning rate: 6.000000E-05 | global batch size:    32 | loss scale: 1073741824.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
 [2024-08-01 09:44:57] iteration        4/       5 | consumed samples:          128 | elapsed time per iteration (ms): 613.5 | throughput per GPU (TFLOP/s/GPU): 1.2 | learning rate: 6.000000E-05 | global batch size:    32 | loss scale: 536870912.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
 [2024-08-01 09:44:58] iteration        5/       5 | consumed samples:          160 | elapsed time per iteration (ms): 624.4 | throughput per GPU (TFLOP/s/GPU): 1.2 | learning rate: 6.000000E-05 | global batch size:    32 | loss scale: 268435456.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
[after training is done] datetime: 2024-08-01 09:44:58
0.02user 0.00system 0:28.09elapsed 0%CPU (0avgtext+0avgdata 30816maxresident)k
0inputs+0outputs (0major+1627minor)pagefaults 0swaps
```

**References:**

1. <https://docs.nvidia.com/nemo-framework/user-guide/latest/getting-started.html>
2. <https://github.com/brevdev/notebooks/blob/main/llama31_law.ipynb>

## 3. Stable Diffusion Web UI

It's best to run `./webui.sh` on a new virtual environment, because `./webui.sh` install lots of
deps from PyPI.

```bash
# See: https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#automatic-installation-on-linux
sudo apt install -V -y libgl1 libglib2.0-0

# Recommended for ./webui.sh. If not done, ./webui.sh will warn.
sudo apt install -V -y bc google-perftools

# Clone the repo. Make sure to do this outside of a git repo (local dir.), otherwise
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Record the commit sha
git branch -vv
# * master 82a973c0 [origin/master] changelog

./webui.sh
# ...
# Running on local URL:  http://127.0.0.1:7860
# ...
```

The default model fits into my GPU.

Test with this prompt:

> a full-body portrait of a kamen rider, standing on a beautiful, green, lush, natural landscape with blue sky and water. The kamen rider must wear the mask.

Leave the settings on their default, but we'll vary the batch hyperparameters.

```text
Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 1741860054,
Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.10.1
```

```console
# Idle
$ gpustat -a -i 2
XXXXXXXXXX                          Fri Aug  2 15:02:15 2024  552.41
[0] NVIDIA RTX A1000 6GB Laptop GPU | 83°C,  ?? %,   0 % (E:   0 %  D:   0 %),    9 /  13 W |  2760 /  6144 MB | xxxxxx:python3.10/389746(?M)

# batch_count=1; batch_size=1
# Time taken: 1 min. 41.5 sec.
# A: 3.10 GB, R: 3.70 GB, Sys: 4.7/6 GB (78.5%)
$ gpustat -a -i 2
XXXXXXXXXX                          Fri Aug  2 15:09:17 2024  552.41
[0] NVIDIA RTX A1000 6GB Laptop GPU | 85°C,  ?? %, 100 % (E:   0 %  D:   0 %),   19 /  20 W |  3902 /  6144 MB | xxxxxx:python3.10/389746(?M)

# batch_count=2; batch_size=1
# Time taken: 3 min. 22.8 sec
# A: 3.10 GB, R: 3.70 GB, Sys: 4.7/6 GB (78.5%)
$ gpustat -a -i 2
XXXXXXXXXX                          Fri Aug  2 15:27:04 2024  552.41
[0] NVIDIA RTX A1000 6GB Laptop GPU | 88°C,  ?? %, 100 % (E:   0 %  D:   0 %),   15 /  13 W |  3912 /  6144 MB | xxxxxx:python3.10/389746(?M)

# batch_count=2; batch_size=2 => Generate 2x2 = 4 images
# Time taken: 6 min. 18.9 sec.
# A: 3.16 GB, R: 4.08 GB, Sys: 5.1/6 GB (84.8%)
$ gpustat -a -i 2
XXXXXXXXXX                          Fri Aug  2 15:30:44 2024  552.41
[0] NVIDIA RTX A1000 6GB Laptop GPU | 84°C,  ?? %, 100 % (E:   0 %  D:   0 %),   13 /  13 W |  4304 /  6144 MB | xxxxxx:python3.10/389746(?M)
```
