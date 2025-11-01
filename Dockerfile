FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.app:app", "--workers", "2"]
