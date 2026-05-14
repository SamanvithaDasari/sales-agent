# Python 3.12 slim base — matches our dev environment
FROM python:3.12-slim

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python deps as root, then switch
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY --chown=user . .

# HF Spaces only allows writes to /tmp — point our data dirs there
ENV SALESAGENT_DATA_DIR=/tmp/salesagent
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Switch to non-root for runtime
USER user

# Port HF Spaces exposes to the internet
EXPOSE 7860

# Startup script: seed DB, build RAG, run both services
CMD ["bash", "start.sh"]
