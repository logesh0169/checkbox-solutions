FROM ubuntu:20.04

ENV TZ=Asia/Kolkata
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN rm -rf *

COPY pyproject.toml /app/
COPY poetry.lock /app/

RUN apt update
RUN apt install -y python3.8
RUN apt install -y python3-pip
RUN pip install poetry
RUN poetry env use python3.8
RUN poetry install
RUN poetry run python -m pip install tensorflow
RUN poetry run python -m pip install keras

COPY src src
COPY utils utils
COPY main.py main.py

RUN pwd

RUN ls -ltr


