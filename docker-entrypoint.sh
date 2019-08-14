#!/bin/bash
ROOT=/home/hsm

cd $ROOT
exec gunicorn -b :8080 --reload --access-logfile - --error-logfile - wsgi:app