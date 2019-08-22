#!/bin/bash
ROOT=/home/hsm/HSM

cd $ROOT
exec gunicorn -b :8080 --timeout 6000000 --reload --access-logfile - --error-logfile - api:app