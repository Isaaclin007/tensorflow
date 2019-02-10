#!/bin/bash

rm -f model/*
python train.py
python test.py

