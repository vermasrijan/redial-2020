#####<HEAVY IMAGE GIVEN BELOW, WITHOUT ROOT USER>#####
# FROM hassanmohsin/rdkit-openbabel:latest
#
# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get update && apt-get install -y libpango1.0-0 \
#     libcairo2 \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*
####################################

#####<USING RDKIT 2019, IN ORDER TO MAKE RDK-DESCRIPTORS WORK>######
FROM informaticsmatters/rdkit-python3-debian:Release_2019_03_1
USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libpango1.0-0 \
    libcairo2 \
    libpq-dev \
    perl \
    && rm -rf /var/lib/apt/lists/*

####################################

#####<LIGHT WEIGHT IMAGE GIVEN BELOW, WITH ROOT USER>#####
# FROM informaticsmatters/rdkit-python3-debian:latest
#
# USER root
# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get update
#
# RUN apt-get install libpango1.0-0 -y
# RUN apt-get install libcairo2 -y
# RUN apt-get install libpq-dev -y
####################################

# MAINTAINER
MAINTAINER Srijan Verma<vermasrijan44@gmail.com>

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY saved_models ./saved_models
COPY static ./static
COPY smi_all_dict.pkl mpro_smi_all_dict.pkl 3cl_smi_all_dict.pkl drug_central_drugs.csv rdkDes_scaler.save volsurf_scaler.save volsurf_scaler.pkl rdkDes_scaler.pkl volsurf_scaler.save run_script.py config.py app.py ./
COPY mayachemtools ./mayachemtools

# For webapp
ENTRYPOINT ["python3", "app.py"]
EXPOSE 5000
