set -ex

for PEN in 1e-04 0.002 0.005 5e-04 0.2 0.5 0.1 0.01 2e-04 0.02 0.05 0.001; do
    Rscript fit-gamm.R $PEN
done
