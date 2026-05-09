FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY genz_classifier/ ./genz_classifier/
COPY setup.py .

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "genz_classifier.main:app", "--host", "0.0.0.0", "--port", "8000"]