FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-sagemaker

RUN pip3 --no-cache-dir install easyocr==1.6.2 sagemaker-inference multi-model-server

COPY ./code/entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /logs
RUN chmod 777 /logs

RUN mkdir -p /home/sbx_user1051/.cache/torch/hub/checkpoints
RUN curl https://johnespe-open-lpr-plate-detection-model.s3.amazonaws.com/resnet50-0676ba61.pth --output /home/sbx_user1051/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth 
RUN chmod 777 /home/sbx_user1051/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

RUN mkdir -p /home/model-server/
ADD ./code /home/model-server/

RUN curl https://johnespe-open-lpr-plate-detection-model.s3.amazonaws.com/SSD.pth --output /home/SSD.pth
RUN chmod 777 /home/SSD.pth

WORKDIR /home/model-server/
RUN python -c 'from standalone import StandAloneInference; StandAloneInference()'

RUN cp -a /root/.cache/ /home/sbx_user1051/.cache/
RUN chmod -R 777 /home/sbx_user1051/.cache

RUN cp -a /root/.EasyOCR/ /home/sbx_user1051/.EasyOCR/
RUN chmod -R 777 /home/sbx_user1051/.EasyOCR

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
