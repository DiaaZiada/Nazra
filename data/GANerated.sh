#!/bin/bash

start=`date +%s`


end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
