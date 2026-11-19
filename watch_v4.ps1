# ---- paths: fill in ----
$repo     = 'C:\Users\uqlcapp1\Simulations\Python'                      # folder you ran python -m fsi.main_sweeps from
$onedrive = "C:\Users\uqlcapp1\OneDrive - The University of Queensland\Documents\Study\PhD\Simulations\Python\results\sweep_v4"    # inside your OneDrive folder -> syncs itself
$rdm      = 'R:\PhD_Lorenzo_Cappello-Q9785\Simulations\Python\results\sweep_v4'            # RDM path as in Explorer (UNC safer than R:)
# -------------------------

$sweep  = Join-Path $repo 'results\sweep_v4'
$log    = Join-Path $repo 'results\sweep_v4_log.txt'
$status = Join-Path $onedrive 'STATUS.txt'
New-Item -ItemType Directory -Force $onedrive | Out-Null
function Write-Status($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  $msg" | Tee-Object -FilePath $status -Append }

# main sweep process (not the worker processes)
$main = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*fsi.main_sweeps*' -and $_.CommandLine -notlike '*multiprocessing*' }
if (-not $main) { Write-Status 'sweep process not found - check the filter'; return }
Write-Status "watching PID $($main.ProcessId)"

# wait; mirror the log to OneDrive every 10 min so progress is visible remotely
while (Get-Process -Id $main.ProcessId -ErrorAction SilentlyContinue) {
    try { Copy-Item $log (Join-Path $onedrive 'sweep_v4_log.txt') -Force -ErrorAction Stop } catch {}
    Start-Sleep -Seconds 600
}
Start-Sleep -Seconds 60
Copy-Item $log (Join-Path $onedrive 'sweep_v4_log.txt') -Force

if (-not (Test-Path (Join-Path $sweep 'sweep_all.npz'))) { Write-Status 'FINISHED WITHOUT sweep_all.npz - see log'; return }
$failed = (Select-String -Path $log -Pattern 'FAILED' | Measure-Object).Count
Write-Status "sweep done, FAILED lines in log: $failed"

# copies (robocopy exit codes < 8 = success)
robocopy $sweep $onedrive /E /R:5 /W:60 /NP /NFL /NDL | Out-Null
Write-Status "OneDrive copy exit code $LASTEXITCODE"
robocopy $sweep $rdm /E /R:5 /W:60 /NP /NFL /NDL | Out-Null
Write-Status "RDM copy exit code $LASTEXITCODE"
Copy-Item $log $rdm -Force

# git: only the small assembled files, never the chunks with profiles
Set-Location $repo
$files = 'results\sweep_v4\sweep_all.npz', 'results\sweep_v4\meta.npz', 'results\sweep_v4_log.txt'
$big = $files | Where-Object { (Get-Item $_).Length -gt 95MB }
if ($big) { Write-Status "git skipped, file > 95 MB: $big"; return }
git add -f -- $files
git commit -m 'Add sweep v4 results (Omega+ = 0.0053, m 1e2-1e4, zeta 0.1-1)' -- $files
git push
Write-Status "git push exit code $LASTEXITCODE"
Write-Status 'ALL DONE'