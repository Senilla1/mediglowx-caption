FROM python:3.10-slim

WORKDIR /app
COPY mgx-clean-http/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# csak a tiszta teszt app kerül a konténerbe
COPY mgx-clean-http/ .

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
