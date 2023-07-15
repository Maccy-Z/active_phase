FROM python:3.10
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update
RUN apt install -y python3-pip git

#RUN pip install scipy
RUN pip install matplotlib numpy==1.23

RUN pip install GPy

RUN apt install -y libgl1
RUN pip install opencv-python
# RUN pip install gpyopt
