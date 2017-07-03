#!/bin/bash

while test -d "/proc/$1"
do
	sleep 5m
done
python main.py data/synth/ >> synth_120_32.txt

