#!/bin/bash

if [[ $OMPI_COMM_WORLD_RANK -gt 0 ]]
then
    exec $@ 1>/dev/null
else
    exec $@ 
fi
