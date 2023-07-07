#!/bin/bash

if [ -z "$1" ]
then
    echo "Missing task"
    exit 1
fi

# ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
# ['morlet', 'morlet2', 'ricker']

./run_train_single.sh "CO(GT)" "morlet" "${1}"
./run_train_single.sh "CO(GT)" "morlet2" "${1}"
./run_train_single.sh "CO(GT)" "ricker" "${1}"

./run_train_single.sh "PT08.S1(CO)" "morlet" "${1}"
./run_train_single.sh "PT08.S1(CO)" "morlet2" "${1}"
./run_train_single.sh "PT08.S1(CO)" "ricker" "${1}"

./run_train_single.sh "C6H6(GT)" "morlet" "${1}"
./run_train_single.sh "C6H6(GT)" "morlet2" "${1}"
./run_train_single.sh "C6H6(GT)" "ricker" "${1}"

./run_train_single.sh "PT08.S2(NMHC)" "morlet" "${1}"
./run_train_single.sh "PT08.S2(NMHC)" "morlet2" "${1}"
./run_train_single.sh "PT08.S2(NMHC)" "ricker" "${1}"

./run_train_single.sh "NOx(GT)" "morlet" "${1}"
./run_train_single.sh "NOx(GT)" "morlet2" "${1}"
./run_train_single.sh "NOx(GT)" "ricker" "${1}"

./run_train_single.sh "PT08.S3(NOx)" "morlet" "${1}"
./run_train_single.sh "PT08.S3(NOx)" "morlet2" "${1}"
./run_train_single.sh "PT08.S3(NOx)" "ricker" "${1}"

./run_train_single.sh "NO2(GT)" "morlet" "${1}"
./run_train_single.sh "NO2(GT)" "morlet2" "${1}"
./run_train_single.sh "NO2(GT)" "ricker" "${1}"

./run_train_single.sh "PT08.S4(NO2)" "morlet" "${1}"
./run_train_single.sh "PT08.S4(NO2)" "morlet2" "${1}"
./run_train_single.sh "PT08.S4(NO2)" "ricker" "${1}"

./run_train_single.sh "PT08.S5(O3)" "morlet" "${1}"
./run_train_single.sh "PT08.S5(O3)" "morlet2" "${1}"
./run_train_single.sh "PT08.S5(O3)" "ricker" "${1}"

./run_train_single.sh "T" "morlet" "${1}"
./run_train_single.sh "T" "morlet2" "${1}"
./run_train_single.sh "T" "ricker" "${1}"

./run_train_single.sh "RH" "morlet" "${1}"
./run_train_single.sh "RH" "morlet2" "${1}"
./run_train_single.sh "RH" "ricker" "${1}"

./run_train_single.sh "AH" "morlet" "${1}"
./run_train_single.sh "AH" "morlet2" "${1}"
./run_train_single.sh "AH" "ricker" "${1}"
