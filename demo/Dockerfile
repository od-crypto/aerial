FROM python:3.7

RUN apt-get -y update && apt-get install -y htop build-essential cmake ncdu lshw unzip wget curl git htop lshw  libglib2.0-0 libsm6 libxext6 libsm6 libxrender1 libfontconfig1

ARG DEVICE=cpu
ENV DEVICE ${DEVICE}

WORKDIR /app

COPY requirements-${DEVICE}.txt app/requirements.txt

RUN pip install -r app/requirements.txt

COPY *.py .

#############################
# streamlit stuff

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
    [general]\n\
    email = \"\"\n\
    " > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
    [server]\n\
    enableCORS = false\n\
    " > /root/.streamlit/config.toml'

#############################

CMD ["streamlit", "run", "app.py"]