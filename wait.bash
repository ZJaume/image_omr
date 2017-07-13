#!/bin/bash

while test -d "/proc/$1"
do
	sleep 5m
done
python main.py $2

