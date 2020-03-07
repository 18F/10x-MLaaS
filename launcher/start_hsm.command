#!/usr/bin/env bash

script_name=launch_server.sh

working_dir=$(dirname "$0")
script_dir=$working_dir/scripts

echo script_dir: $script_dir

open -a Terminal.app $script_dir/$script_name
