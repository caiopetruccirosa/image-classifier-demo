FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential curl software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# To install opencv-python-headless and its dependencies
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 libgl1 -y

COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py"]