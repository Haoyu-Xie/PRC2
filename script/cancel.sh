#!/bin/bash

for ((i = $1; i <= $2; i++)); do
    scancel $i
done
