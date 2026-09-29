Write-Host "Running Phase 2A 20-Run Grid..."

Write-Host "Running Baseline 0 (Classifier Only)..."
python experiments/exp_classifier_only.py > logs_baseline_0.txt

Write-Host "Running Baseline 1 (Fixed Kernel)..."
python experiments/exp_fixed_kernel.py > logs_baseline_1.txt

Write-Host "Running Main Model (Trainable Kernel)..."
python experiments/exp_trainable_kernel.py > logs_trainable.txt

Write-Host "Running Ablation (Trainable Seed)..."
python experiments/exp_kernel_ablation.py > logs_ablation.txt

Write-Host "Grid complete! Generating Final Phase 2A Decision Report..."
python generate_phase2a_report.py
