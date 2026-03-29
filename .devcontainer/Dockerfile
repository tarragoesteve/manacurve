FROM mcr.microsoft.com/devcontainers/python:3.12

# Accept proxy settings as build-time arguments so apt-get can reach the network
# when building behind a corporate proxy.
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Clear proxy env vars from the final image layer so they aren't baked in.
ENV http_proxy= \
    https_proxy= \
    no_proxy=
