FROM python:3.12

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY . .

RUN pip install -r requirements.txt

# Ensure expected writable directories exist (for local Qdrant + uploads).
RUN mkdir -p uploaded_docs qdrant_storage \
	&& chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]