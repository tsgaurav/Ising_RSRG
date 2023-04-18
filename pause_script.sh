#!/bin/bash

for pid in $(pgrep python); do kill -TSTP $pid; done
