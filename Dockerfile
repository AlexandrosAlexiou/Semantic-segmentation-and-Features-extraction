FROM tensorflow/tensorflow:1.8.0-py3

WORKDIR /code

COPY requirements.txt /code

RUN pip install --upgrade pip && pip install -r requirements.txt

ADD . /code