#!/bin/bash

for pid in $(pgrep python); do kill -CONT $pid; done
