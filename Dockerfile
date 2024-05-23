FROM continuumio/anaconda3:latest
LABEL authors="abrahamalbert"
# Copy files
COPY . /home/Projects/ForeverDreaming
WORKDIR /home/Projects/ForeverDreaming

# Install pytorch with CPU
RUN conda install pytorch cpuonly -c pytorch
RUN conda install transformers
RUN conda install dash
#ENTRYPOINT ["bash"]