FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
#RUN apt-get install -y xserver-xorg
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz \
      && tar Jxvf ./ffmpeg-git-amd64-static.tar.xz \
      && cp ./ffmpeg-git-*-amd64-static/ffmpeg /usr/local/bin/

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy
RUN pip install pandas
RUN pip install opencv-python
#ADD ./src /root/opt