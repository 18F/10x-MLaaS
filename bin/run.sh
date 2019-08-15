#!/bin/bash
ROOT=HSM

cd $ROOT
gunicorn api:app