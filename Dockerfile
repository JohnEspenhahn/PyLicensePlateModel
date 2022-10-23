FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-sagemaker

RUN pip3 --no-cache-dir install easyocr==1.6.2 sagemaker-inference multi-model-server

COPY ./code/entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/
ADD ./code /home/model-server/
ADD ./SSD.pth /home/model-server/

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
