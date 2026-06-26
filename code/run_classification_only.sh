#!/bin/bash
set -e

# Run all commands from the repository root (parent of code/)
cd "$(dirname "$0")/.."

echo "=========================================="
echo "1/4: Running ablation study for WavLM Base-Plus"
echo "=========================================="
python3 -u code/classification/run_comprehensive_ablation.py --model base-plus

echo "=========================================="
echo "2/4: Running ablation study for WavLM Large"
echo "=========================================="
python3 -u code/classification/run_comprehensive_ablation.py --model large

echo "=========================================="
echo "3/4: Generating final comparison report"
echo "=========================================="
python3 -u code/classification/compare_results.py

echo "=========================================="
echo "4/4: Computing 95% Confidence Intervals (Bootstrap N=2000)"
echo "=========================================="
python3 -u code/classification/compute_bootstrap_ci.py > output/bootstrap_ci_tables.md
echo "CI tables saved to output/bootstrap_ci_tables.md"

echo "=========================================="
echo "REMAINING PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
