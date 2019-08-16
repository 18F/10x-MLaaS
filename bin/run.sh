#!/bin/bash
ROOT=$HOME/HSM

cd $ROOT
gunicorn api:app