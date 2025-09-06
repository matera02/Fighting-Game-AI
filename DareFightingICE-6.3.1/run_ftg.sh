#!/bin/bash

# Number of rounds (default: 100 if not specified)
REPEATS=${1:-100}

# Command to start FightingICE with a specific number of rounds
java -cp FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/* \
     Main --limithp 400 400 --grey-bg -r "$REPEATS"

