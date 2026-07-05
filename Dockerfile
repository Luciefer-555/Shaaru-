FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for colour-science and opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
