#!/bin/bash
echo "Starting MLaaS" && cd /home/hsm && pip install -r HSM/requirements.txt && su - hsm
/bin/bash