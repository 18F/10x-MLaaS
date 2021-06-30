FROM python:3.9.6

RUN apt-get update && apt-get install -y postgresql-client

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pip install pipenv
RUN pipenv install --system --dev

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Add hsm user
RUN useradd hsm && echo "hsm:hsm" | chpasswd && adduser hsm sudo

EXPOSE 8080
