FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --user -r requirements.txt

ENV PATH=/root/.local/bin:$PATH

COPY /PDF_data /app/PDF_data
COPY /utils /app/utils
COPY notification.txt /app/notification.txt
COPY main.py /app/main.py

EXPOSE 7860

CMD ["python", "main.py"]

# docker build -t <image's name>:<image's version> .