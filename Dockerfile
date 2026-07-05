FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for colour-science and opencv
# NOTE: libgl1-mesa-glx was renamed to libgl1 in Debian Bullseye+
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
