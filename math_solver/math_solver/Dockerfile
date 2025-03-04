FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --user -r requirements.txt

ENV PATH=/root/.local/bin:$PATH

COPY main.py /app/main.py

EXPOSE 7861

CMD ["python", "main.py"]