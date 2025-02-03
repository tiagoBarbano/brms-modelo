FROM python:3.13-slim

ENV AWS_S3_BUCKET_NAME=my-bucket
ENV AWS_REGION=ca-central-1
ENV MONGO_PASSWORD=
ENV MONGO_USER=mongodb:27017
ENV MONGO_HOST=localhost:27017
ENV MONGO_READ_PREFERENCE=
ENV COLLECTION_MODEL=modelos
ENV COLLECTION_HISTORICO_MODEL=historico_modelos
ENV DATABASE_MONGODB=modelagem

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY app.py .
COPY aws_utils.py .
COPY config.py .
COPY database.py .
COPY utils.py .
COPY model.py .
COPY repository_historico_modelo.py .
COPY repository_modelos.py .
COPY requirements.txt .

RUN mkdir images
COPY images/brms-logo.png images/brms-logo.png
RUN mkdir modelos_custom
RUN mkdir modelos_historicos
RUN mkdir upload_bases
RUN mkdir data

RUN pip install -r requirements.txt

EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]