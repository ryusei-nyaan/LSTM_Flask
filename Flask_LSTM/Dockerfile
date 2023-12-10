FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip

COPY ./flasktestLSTM/requirements.txt /flasktestLSTM/
COPY ./flasktestLSTM/RNNPredict.py /flasktestLSTM/
COPY ./flasktestLSTM/server_RNN.py /flasktestLSTM/
COPY ./flasktestLSTM/wsgi.py /flasktestLSTM/
COPY ./flasktestLSTM/templates/index2.html /flasktestLSTM/templates/
COPY ./flasktestLSTM/templates/result_2.html /flasktestLSTM/templates/
COPY ./flasktestLSTM/app.ini /flasktestLSTM/
WORKDIR /flasktestLSTM/
RUN mkdir static
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD ["uwsgi", "--ini", "app.ini"]
