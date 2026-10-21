FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake make openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
