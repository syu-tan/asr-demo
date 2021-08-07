FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt update
RUN apt install -y sox

# Copy local code to the container image.
ENV APP_HOME /asr
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Run the web service on container startup. 
EXPOSE $PORT
ENTRYPOINT [ "sh", "entrypoint.sh" ] 