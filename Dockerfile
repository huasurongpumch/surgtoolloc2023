ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update 
RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev 
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*

# Install MMEngine , MMCV and MMDet
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0rc6,<3.1.0"

# Install MMYOLO
COPY mmyolo-0.5.0 /mmyolo
WORKDIR /mmyolo
RUN mim install --no-cache-dir -e .
WORKDIR /mmyolo/mmdetection-3.0.0
RUN pip install --no-cache-dir -e .
WORKDIR /mmyolo/mmpretrain-1.0.1
RUN mim install --no-cache-dir -e .
COPY OSTrack-main_surg_docker /OSTrack
WORKDIR /OSTrack
RUN bash install.sh
RUN python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
WORKDIR /mmyolo

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

#RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user -rrequirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user numba
COPY --chown=algorithm:algorithm process.py /opt/algorithm/process.py

ENTRYPOINT python -m process $0 $@
