# ========== build the go app ==========
FROM golang:bullseye AS builder
WORKDIR /build

# system update and dependencies installation
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential wget ca-certificates

COPY ./app .
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o main .

# download the ONNX runtime library
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz && \
    tar -xvf onnxruntime-linux-x64-1.15.0.tgz && \
    mv onnxruntime-linux-x64-1.15.0/lib/libonnxruntime.so.1.15.0 . && \
    rm -rf onnxruntime-linux-x64-1.15.0* # clean up the tar file

# download the ONNX model
RUN wget https://github.com/ziyixi/PhaseNet-TF/releases/download/v0.3.0/model.onnx

# ========== runtime image ==========
FROM debian:bullseye-slim
LABEL org.opencontainers.image.authors="docker@ziyixi.science"
LABEL org.opencontainers.image.source=https://github.com/ziyixi/PhaseNet-TF
LABEL org.opencontainers.image.description="PhaseNet-TF: Advanced Seismic Arrival Time Detection via Deep Neural Networks in the Spectrogram Domain, Leveraging Cutting-Edge Image Segmentation Approaches"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /app

# Copy binary and necessary files from build stage
COPY --from=builder /build/main /app/main
COPY --from=builder /build/libonnxruntime.so.1.15.0 /app/
COPY --from=builder /build/model.onnx /app/

ENV onnx_lib /app/libonnxruntime.so.1.15.0
ENV onnx_model /app/model.onnx
ENV PORT 8080

CMD ["./main"]
