set -ex

mkdir -p logs checkpoints

for e in 1 2 3 4; do
for p in 1 2 5; do
for i in 0..9; do
PEN=${p}e-${e}
CKP="checkpoints/testpreds_fold=${i}_pen=$PEN.pt"
if [[ -e "$CKP" ]]; then
    echo "checkpoint found" \$CKP
    exit 0
fi

python run.py train $i 10 --randef-penalty $PEN
done
done
done
