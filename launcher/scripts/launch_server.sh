#!/usr/bin/env bash

working_dir=$(dirname "$0")
cd $working_dir
cd ..
cd ..

docker-compose up --build

