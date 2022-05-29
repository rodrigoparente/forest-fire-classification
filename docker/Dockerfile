# Example of command to build image and push
# docker build --build-arg CODE_DIR=tflite-code --build-arg MODEL_FILE=model_mobilenet_lite.tflite --build-arg MODEL_TYPE=tflite --file ./docker/Dockerfile -t ff-classify:aws-mobilenet-lite .
# aws ecr get-login-password --region <regiao> | docker login --username AWS --password-stdin <conta>.dkr.ecr.<regiao>.amazonaws.com
# docker tag ff-classify:aws-mobilenet-lite <conta>.dkr.ecr.<regiao>.amazonaws.com/ff-classify:aws-mobilenet-lite
# docker push <conta>.dkr.ecr.us-east-1.amazonaws.com/ff-classify:aws-mobilenet-lite

FROM python:3.8-slim-buster

ARG CODE_DIR
ARG MODEL_FILE
ARG MODEL_TYPE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /project

# Install aws-lambda-cpp build dependencies
RUN apt-get update \
  && apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

COPY ${CODE_DIR}/requirements.txt  .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install awslambdaric boto3

COPY models/${MODEL_FILE} model.${MODEL_TYPE}

COPY ${CODE_DIR}/classify.py .

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "classify.lambda_handler" ]
