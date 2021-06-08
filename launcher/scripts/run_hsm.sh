#!/usr/bin/env bash

docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} "/bin/bash" -c "python ~/HSM/app.py"

