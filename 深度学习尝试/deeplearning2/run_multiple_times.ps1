# PowerShell script to run training 100 times with different output directories

for ($i=1; $i -le 1000; $i++) {
    # Create output directory with absolute path
    $outputDir = Join-Path "D:\train_runs" "run_$($i.ToString('000'))"
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    if (-not (Test-Path -Path $outputDir -PathType Container)) {
          Write-Error "Failed to create directory: $outputDir"
          continue
      }
    Write-Host "Starting training run $i in directory: $outputDir"

    # Run training with specified output directory
    c:\users\19057\appdata\local\programs\python\python39\python.exe train.py --output_dir "$outputDir"
    $trainExitCode = $LASTEXITCODE

    # Check if training succeeded and log file was created
    $logPath = [System.IO.Path]::GetFullPath((Join-Path $outputDir "training_log.csv"))
    $outputDirAbs = [System.IO.Path]::GetFullPath($outputDir)
    if ($trainExitCode -ne 0) {
        Write-Host "Error: Training failed with exit code $trainExitCode in $outputDir. Skipping analysis."
        continue
    }
    if (-not (Test-Path $logPath)) {
        Write-Host "Error: Training log not found in $outputDir. Skipping analysis."
        continue
    }

    # Run analysis and save plot in the same directory
    python analysis.py --input_log "$logPath" --output_dir "$outputDirAbs"

    Write-Host "Completed training run $i"
}

Write-Host "All 1000 training runs completed successfully!"