FROM alpine AS model

## Setup models for hent-AI
WORKDIR /model

# Download models.
RUN wget https://github.com/erogaki-dev/erogaki-models/releases/download/v0.0.1/hent-AI.model.268.tar.gz
RUN wget https://github.com/erogaki-dev/erogaki-models/releases/download/v0.0.1/25-11-2019.Fatal.Pixels.tar.gz

# Unpack models.
RUN tar -xf hent-AI.model.268.tar.gz
RUN tar -xf 25-11-2019.Fatal.Pixels.tar.gz

FROM continuumio/miniconda3 AS deploy

## Copy models.
WORKDIR /model

COPY --from=model ["/model/hent-AI model 268", "./hent-AI model 268"]
COPY --from=model ["/model/25-11-2019 Fatal Pixels", "./25-11-2019 Fatal Pixels"]

## Setup first part of hent-AI-erogaki-wrapper.
WORKDIR /app

# Create the conda environment.
RUN conda create --name hent-AI-erogaki-wrapper python=3.5.2 --channel conda-forge

## Setup hent-AI.
WORKDIR /app/hent-AI

# Copy the requirements file.
COPY ./hent-AI/requirements-cpu.txt ./

# Install the dependencies.
RUN conda run --no-capture-output --name hent-AI-erogaki-wrapper pip install -r requirements-cpu.txt

# Install libgl.
RUN apt-get install -y libgl1-mesa-glx

# Copy the source code.
COPY ./hent-AI ./

# Do: "6. Install Mask Rcnn"
RUN conda run --no-capture-output --name hent-AI-erogaki-wrapper python setup.py install

## Setup second part hent-AI-erogaki-wrapper.
WORKDIR /app

# Copy the requirements file.
COPY ./src/requirements.txt ./

# Install the dependencies.
RUN conda run --no-capture-output --name hent-AI-erogaki-wrapper pip install -r requirements.txt

# Copy the source code.
COPY ./src ./

# Start hent-AI-erogaki-wrapper.
ENTRYPOINT ["conda", "run", "--no-capture-output", "--name", "hent-AI-erogaki-wrapper", "python", "wrapper_main.py"]
