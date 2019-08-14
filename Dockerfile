FROM python:3.6.8


RUN apt-get update && apt-get install -y postgresql-client

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pip install pipenv
RUN pipenv install --system --dev

# COPY HSM HSM
# COPY wsgi.py docker-entrypoint.sh ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh


# Add hsm user
RUN useradd hsm && echo "hsm:hsm" | chpasswd && adduser hsm sudo

# RUN chown -R hsm:hsm ./
# USER hsm

EXPOSE 8080

# ENTRYPOINT ["./docker-entrypoint.sh]
