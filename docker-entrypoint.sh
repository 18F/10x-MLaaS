#!/bin/bash
ROOT=/home/hsm/HSM

cd $ROOT
exec gunicorn -b :$HOST_PORT --reload --access-logfile - --error-logfile - api:app