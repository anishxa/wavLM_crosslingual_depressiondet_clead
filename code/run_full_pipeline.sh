#!/bin/bash
set -e

# Run all commands from the repository root (parent of code/)
cd "$(dirname "$0")/.."

echo "=========================================="
echo "1/6: Extracting features for WavLM Base-Plus"
echo "=========================================="
python3 -u code/feature_extraction/extract_ablation_features.py --model base-plus --device auto --batch_size 8

echo "=========================================="
echo "2/6: Extracting features for WavLM Large"
echo "=========================================="
python3 -u code/feature_extraction/extract_ablation_features.py --model large --device auto --batch_size 8

echo "=========================================="
echo "3/6: Running ablation study for WavLM Base-Plus"
echo "=========================================="
python3 -u code/classification/run_comprehensive_ablation.py --model base-plus

echo "=========================================="
echo "4/6: Running ablation study for WavLM Large"
echo "=========================================="
python3 -u code/classification/run_comprehensive_ablation.py --model large

echo "=========================================="
echo "5/6: Generating final comparison report"
echo "=========================================="
python3 -u code/classification/compare_results.py

echo "=========================================="
echo "6/6: Computing 95% Confidence Intervals (Bootstrap N=2000)"
echo "=========================================="
python3 -u code/classification/compute_bootstrap_ci.py > output/bootstrap_ci_tables.md
echo "CI tables saved to output/bootstrap_ci_tables.md"

echo "=========================================="
echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
