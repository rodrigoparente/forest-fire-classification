# Example of command to build image, test and push
# docker build --build-arg CODE_DIR=tf-code --build-arg MODEL_FILE=model_mobilenet.h5 --build-arg MODEL_TYPE=h5 --file ./docker/test/Dockerfile -t ff-classify:mobilenet .
# docker run ff-classify:mobilenet
# docker login
# docker tag ff-classify:mobilenet sayonarasantos/ff-classify:mobilenet
# docker push sayonarasantos/ff-classify:tf

FROM python:3.8-slim-buster

ARG CODE_DIR
ARG MODEL_FILE
ARG MODEL_TYPE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /project

COPY ${CODE_DIR}/requirements.txt  .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY models/${MODEL_FILE} model.${MODEL_TYPE}

COPY ${CODE_DIR}/local-test-classify.py .
COPY docker/test/*.jpg ./

CMD ["python", "local-test-classify.py"]