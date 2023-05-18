FROM python:3.8-slim-buster

WORKDIR /app

RUN pip install \
  pylint \
  pylint-flask \
  pytest \
  pytest-cov

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
