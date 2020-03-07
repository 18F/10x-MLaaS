#!/usr/bin/env bash

script_dir=$(dirname "$0")
echo script_dir: $script_dir
cd $script_dir
cd ..
cd ..

working_dir=$(dirname "$0")
echo working_dir: $( pwd )

#pipenv shell
pipenv run $script_dir/run_hsm.sh
#pipenv run echo "name: ${CONTAINER_NAME_WEB}"
#echo "name: ${CONTAINER_NAME_WEB}"
# pipenv run docker exec --user hsm --workdir /home/hsm -it ${CONTAINER_NAME_WEB} "/bin/bash" -c "python ~/HSM/app.py"
