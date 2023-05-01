FROM python:3.9-slim

RUN apt-get update && apt-get install -y wget

RUN mkdir /bda-stackoverflow-tags
WORKDIR /bda-stackoverflow-tags

COPY requirements.txt .
COPY pyspark-job.py .

COPY submit-pyskark-job.py .
COPY bdastackoverflow-gcp-sa-key.json .

ENV GOOGLE_APPLICATION_CREDENTIALS="/bda-stackoverflow-tags/bdastackoverflow-gcp-sa-key.json"

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "submit-pyskark-job.py", "--pyspark_file", "/bda-stackoverflow-tags/pyspark-job.py"]
