FROM python:3.7

# Copy local code to the container image.
#  ENV APP_HOME /app
#  WORKDIR $APP_HOME
#  COPY . .

LABEL maintainer = dseneh@gmail.com

# RUN apt-get update -y && \
#    apt-get install -y python-pip python-dev

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8080
ENTRYPOINT [ "python" ]
# Install production dependencies.
# RUN pip install -r requirements.txt 
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD exec gunicorn — bind :$PORT — workers 1 — threads 8 app:app
CMD ["app.py" ]