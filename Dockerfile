FROM python:3.13-alpine3.22

WORKDIR /app

RUN apk update \
	&& apk upgrade \
	&& rm -rf /var/cache/apk/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
