FROM python:3.8.12-slim
LABEL maintainer="faisal.ajmal@gmail.com"


RUN DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget curl

RUN pip3 install \
    torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
    
RUN pip3 install easyocr==1.6.2

RUN pip3 install multi-model-server sagemaker-inference

WORKDIR /openlpr

ADD ./code /openlpr

ENTRYPOINT  [ "python3", "/openlpr/entrypoint.py"]
