#! /bin/bash
WHAT=$1
DIR=$2
echo $WHAT script generation
cat $WHAT.hh > $WHAT.gnuplot

DATA_FILE=`find $DIR -name "*.dat" | gr