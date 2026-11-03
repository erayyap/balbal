# balbal server image: OpenAI-compatible HTTP API + live WebSocket on port 8000, or Home Assistant (Wyoming) on port 10300.
#   docker build -t balbal .                                         # from PyPI; --build-arg PRELOAD=balbal-v2 puts the model in the image
#   docker run -p 8000:8000 -v balbal-models:/models balbal           # balbal serve; or: ... balbal serve --model manas --api-key KEY
#   docker run -p 10300:10300 -v balbal-models:/models balbal wyoming
# The model is downloaded into /models on first start unless it was preloaded.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 balbal && mkdir /models && chown balbal /models
# BALBAL: what pip installs, e.g. "balbal[server,wyoming]==0.2.0" or a wheel file in the build context ("/ctx/balbal-....whl[server,wyoming]")
ARG BALBAL="balbal[server,wyoming]"
RUN --mount=type=bind,source=.,target=/ctx pip install --no-cache-dir "$BALBAL"
USER balbal
ENV BALBAL_CACHE=/models BALBAL_HOST=0.0.0.0
ARG PRELOAD=""
RUN for m in $PRELOAD; do balbal download --model "$m"; done
EXPOSE 8000 10300
ENTRYPOINT ["balbal"]
CMD ["serve"]
