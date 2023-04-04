FROM python:3.9-slim

RUN apt-get update && apt-get install -y wget
RUN wget https://dl.google.com/cloudagents/add-dataproc-agent-repo.sh
RUN bash add-dataproc-agent-repo.sh
RUN apt-get update && apt-get install -y google-cloud-sdk google-cloud-sdk-dataproc

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY my_pyspark_job.py .
COPY submit_pyspark_job.py .

CMD ["python", "submit_pyspark_job.py"]
