FROM tensorflow/tensorflow:1.13.1-gpu-jupyter
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update && apt install -y wget git vim
WORKDIR /home
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh && \
    wget -O env.yml "https://drive.google.com/uc?export=download&id=1JgCV804yc7FfOoP41NbPJFEVu4IH5J9k" && \
    mkdir /home/deepqueuenet
    # git clone https://github.com/CarlosGuemS/deepqueuenet.git && \
    # wget https://www.dropbox.com/s/q56sx4hxe93n4g5/DeepQueueNet-dataset.zip
# RUN unzip DeepQueueNet-dataset.zip && \
#     mv DeepQueueNet-synthetic\ data/* deepqueuenet && \
#     rm -rf DeepQueueNet-synthetic\ data DeepQueueNet-dataset.zip  __MACOSX
RUN bash Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -b -p /home/miniconda3
RUN rm Miniconda3-py37_23.1.0-1-Linux-x86_64.sh && /home/miniconda3/bin/conda init bash
RUN /home/miniconda3/bin/conda env create -f env.yml
RUN mkdir /home/deepqueuenet/data
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib/x86_64-linux-gnu"
EXPOSE 8888
CMD ["/bin/bash "]