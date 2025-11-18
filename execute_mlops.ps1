# ================================
# MLOps Pipeline Runner (PS 5.1)
# ================================
Write-Host "=== Starting MLOps Pipeline ===" -ForegroundColor Cyan

# -------------------------------
# CHECK PYTHON
# -------------------------------
Write-Host "Checking Python..."
try { python --version } catch {
    Write-Host "Python not found." -ForegroundColor Red
    exit 1
}

# -------------------------------
# CHECK GIT
# -------------------------------
Write-Host "Checking Git..."
try { git --version } catch {
    Write-Host "Git not found." -ForegroundColor Red
    exit 1
}

# -------------------------------
# CHECK DVC
# -------------------------------
Write-Host "Checking DVC..."
try { python -m dvc --version } catch {
    Write-Host "DVC not found." -ForegroundColor Red
    exit 1
}

# -------------------------------
# CONFIGURE DVC S3 REMOTE (IF NOT ALREADY SET)
# -------------------------------
Write-Host "Configuring DVC S3 remote..."
$RemoteName = "origin"
if (!(python -m dvc remote list | Select-String $RemoteName)) {
    python -m dvc remote add -d $RemoteName s3://sowmya-dvc-store
    python -m dvc remote modify $RemoteName region ap-south-1
    Write-Host "DVC S3 remote configured: s3://sowmya-dvc-store (ap-south-1)."
} else {
    Write-Host "DVC remote '$RemoteName' already exists."
}

# =======================================================
# AUTO-MOVE ONLY MODEL ARTIFACT FILES TO models/ FOLDER
# (NOT environment DLLs or DVC cache)
# =======================================================
Write-Host "`nScanning for large ML model artifacts..."
# Valid model artifact extensions
$AllowedExt = @(".pkl", ".joblib", ".model", ".bin", ".h5", ".pt")
# Forbidden location keywords (must NOT be moved)
$BlockedFolders = @(
    ".venv",
    "venv",
    "env",
    "site-packages",
    ".dvc\cache",
    "AppData"
)
# Find large files (>50MB)
$LargeFiles = Get-ChildItem -Recurse -File | Where-Object {
    $_.Length -gt 50MB -and
    ($AllowedExt -contains $_.Extension) -and
    ($BlockedFolders -notcontains ($_.FullName.Split('\')[3]))
}
if ($LargeFiles.Count -gt 0) {
    Write-Host "Large ML model files found. Moving into models/ safely..."
    if (!(Test-Path "models")) {
        New-Item -Path "models" -ItemType Directory | Out-Null
    }
    foreach ($file in $LargeFiles) {
        Write-Host "Moving: $($file.FullName)"
        Move-Item $file.FullName "models/" -Force
    }
} else {
    Write-Host "No model artifacts need moving."
}

# -------------------------------
# ENSURE .GITIGNORE EXCLUDES LARGE FILES/DIRS (FOR GIT)
# -------------------------------
Write-Host "`nUpdating .gitignore to exclude large files and dirs..." -ForegroundColor Green
$GitIgnoreContent = @"
# Large data and models (handled by DVC/S3)
data/
models/
mlruns/
.venv/
venv/
env/

# Large binaries
*.dll
*.pkl
*.joblib
*.h5
*.pt
*.bin

# OS generated files
.DS_Store
Thumbs.db

# Logs
*.log
"@
$GitIgnoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8 -Force
Write-Host ".gitignore updated."

# Remove large dirs/files from Git index (if tracked)
Write-Host "Removing large files/dirs from Git index..." -ForegroundColor Yellow
git rm -r --cached data/ 2>$null | Out-Null
git rm -r --cached models/ 2>$null | Out-Null
git rm -r --cached mlruns/ 2>$null | Out-Null
git rm -r --cached .venv/ 2>$null | Out-Null
git rm --cached "*.dll" 2>$null | Out-Null
git rm --cached "*.csv" 2>$null | Out-Null  # Specific to data CSVs
Write-Host "Large files removed from Git index."

# -------------------------------
# RUN PYTHON PREP
# -------------------------------
Write-Host "`nRunning data preparation..." -ForegroundColor Yellow
python src/data_prep.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "data_prep.py failed." -ForegroundColor Red
    exit 1
}

# -------------------------------
# RUN TRAINING
# -------------------------------
Write-Host "`nRunning model training..." -ForegroundColor Yellow
python src/train.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "train.py failed." -ForegroundColor Red
    exit 1
}

# -------------------------------
# RUN DVC REPRO
# -------------------------------
Write-Host "`nRunning DVC pipeline using dvc repro..." -ForegroundColor Green
python -m dvc repro
if ($LASTEXITCODE -ne 0) {
    Write-Host "DVC pipeline failed." -ForegroundColor Red
    exit 1
}

# -------------------------------
# DVC PUSH TO S3 (UPDATE DATA FILES AFTER PREP/TRAINING)
# -------------------------------
Write-Host "`nPushing DVC-tracked data and models to S3 remote..." -ForegroundColor Green
try {
    python -m dvc push
    if ($LASTEXITCODE -ne 0) {
        Write-Host "DVC push failed. Attempting to correct by pulling latest and re-adding tracked files..." -ForegroundColor Yellow
        python -m dvc pull  # Pull latest from S3 to sync
        git add .dvc/  # Re-add DVC metadata
        $TrackedFiles = (python -m dvc list -R data/ models/).Trim()
        if ($TrackedFiles) {
            python -m dvc add $TrackedFiles  # Re-add any new/changed data/model files
        }
        python -m dvc push  # Retry push
        if ($LASTEXITCODE -ne 0) {
            Write-Host "DVC push correction failed. Exiting." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "DVC push to S3 completed successfully." -ForegroundColor Green
} catch {
    Write-Host "Unexpected error during DVC push. Exiting." -ForegroundColor Red
    exit 1
}

# -------------------------------
# GIT COMMIT AND PUSH (CODE AND METADATA ONLY)
# -------------------------------
Write-Host "`nUpdating Git repository (code and metadata only)..." -ForegroundColor Green
try {
    git add .gitignore  # Add updated gitignore
    git add .  # Add code, DVC files, metrics, etc. (large files excluded)
    $CommitOutput = git commit -m "Auto-update after MLOps pipeline: data prep, training, and DVC repro"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "No changes to commit. Skipping Git commit." -ForegroundColor Yellow
    } else {
        Write-Host "Changes committed."
    }
    # Check if upstream branch is set
    $UpstreamCheck = git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
    if ($LASTEXITCODE -ne 0 -or !$UpstreamCheck) {
        Write-Host "No upstream branch set. Pushing with --set-upstream..." -ForegroundColor Yellow
        git push --set-upstream origin main
    } else {
        git push
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Git push failed. Attempting force push with lease..." -ForegroundColor Yellow
        if ($LASTEXITCODE -ne 0 -or !$UpstreamCheck) {
            git push --set-upstream origin main --force-with-lease
        } else {
            git push --force-with-lease
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Git force push failed. Exiting." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "Git repository updated successfully (code and metadata only)." -ForegroundColor Green
} catch {
    Write-Host "Unexpected error during Git update. Exiting." -ForegroundColor Red
    exit 1
}

# -------------------------------
# LAUNCH MLflow
# -------------------------------
Write-Host "`nStarting MLflow UI in new window..."
Start-Process powershell -ArgumentList "mlflow ui --port 5000"

# -------------------------------
# COMPLETE
# -------------------------------
Write-Host "`n=== MLOps Pipeline Completed Successfully ===" -ForegroundColor Cyan

# -------------------------------
# FINALLY RUN APP.PY
# -------------------------------
Write-Host "`nRunning src/app.py..." -ForegroundColor Yellow
python src/app.py