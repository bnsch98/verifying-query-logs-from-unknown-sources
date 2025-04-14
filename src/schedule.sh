#!/bin/bash


RAY_ADDRESS="" python thesis_schneg/cluster.py aol 75 100
echo "######### AOL Finished #########"
sleep 10
RAY_ADDRESS="" python thesis_schneg/cluster.py aql 75 100
echo "######### AQL Finished #########"
sleep 10
RAY_ADDRESS="" python thesis_schneg/cluster.py orcas 75 100
echo "######### ORCAS Finished #########"
sleep 10
RAY_ADDRESS="" python thesis_schneg/cluster.py ms-marco 75 100
echo "######### MS MARCO Finished #########"

