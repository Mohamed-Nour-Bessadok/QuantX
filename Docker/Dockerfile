FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libomp-dev

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENTRYPOINT ["sh", "Docker/entrypoint.sh"]
