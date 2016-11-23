#!/bin/bash

gcc driver.c mat_vec.c -o driver -lm
gcc driver_gsl.c mat_vec_gsl.c -o driver_gsl -lgsl -lgslcblas -lm