# syntax=docker/dockerfile:1.7

FROM maven:3.9.11-eclipse-temurin-17 AS builder
WORKDIR /build

COPY pom.xml ./
COPY src ./src

RUN --mount=type=cache,target=/root/.m2 \
    mvn -B -DskipTests clean package

FROM ubuntu:24.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG FIJI_URL=https://downloads.imagej.net/fiji/latest/fiji-linux64.zip

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    openjdk-17-jre \
    tini \
    unzip \
    wget \
    xauth \
    xvfb \
    libfontconfig1 \
    libfreetype6 \
    libglu1-mesa \
    libgtk-3-0 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/fiji.zip "$FIJI_URL" \
    && unzip -q /tmp/fiji.zip -d /opt \
    && rm -f /tmp/fiji.zip \
    && chmod +x /opt/Fiji.app/ImageJ-linux64

COPY --from=builder /build/target/ST2WSI_Registration-*.jar /opt/Fiji.app/plugins/
COPY run_st2wsi.sh /usr/local/bin/st2wsi-cli
COPY docker-entrypoint.sh /usr/local/bin/st2wsi

RUN chmod +x /usr/local/bin/st2wsi /usr/local/bin/st2wsi-cli

ENV FIJI_PATH=/opt/Fiji.app/ImageJ-linux64
ENV FIJI_BIN=/opt/Fiji.app/ImageJ-linux64

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/st2wsi"]
CMD ["help"]
