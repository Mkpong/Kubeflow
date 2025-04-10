FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY train.py .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install torch-model-archiver

CMD ["python", "train.py"]
