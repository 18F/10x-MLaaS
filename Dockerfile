# Base DockerFile for GSA 10x HSM
FROM ubuntu:17.10

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git
RUN apt-get install -y zip
RUN apt-get install -y postgresql-client

# update pip
RUN python3.6 -m pip install --upgrade pip==9.0.3
RUN python3.6 -m pip install wheel virtualenv
RUN cd /usr/local/bin && ln -s /usr/bin/python3 python

# Add hsm user
RUN useradd -m hsm && echo "hsm:hsm" | chpasswd && adduser hsm sudo
USER hsm
WORKDIR /home/hsm

ENTRYPOINT /bin/bash

EXPOSE 5000