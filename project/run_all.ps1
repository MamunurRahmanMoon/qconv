Write-Host "Running exp_classifier_only..."
python experiments/exp_classifier_only.py > results_0_classifier.txt
Write-Host "Running exp_fixed_kernel..."
python experiments/exp_fixed_kernel.py > results_1_fixed.txt
Write-Host "Running exp_trainable_kernel..."
python experiments/exp_trainable_kernel.py > results_2_trainable.txt
Write-Host "Running exp_kernel_ablation..."
python experiments/exp_kernel_ablation.py > results_3_ablation.txt
Write-Host "All experiments completed."
