param(
    [string]$LogDir = "logs",
    [int]$Port = 6006
)

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Get-Process tensorboard -ErrorAction SilentlyContinue | Stop-Process
}

$tbProcess = Start-Process -FilePath "tensorboard" -ArgumentList "--logdir", $LogDir, "--port", $Port -PassThru
try {
    Write-Host ("TensorBoard: http://127.0.0.1:{0}" -f $Port)
    Wait-Process -Id $tbProcess.Id
} finally {
    if ($tbProcess) {
        Stop-Process -Id $tbProcess.Id -ErrorAction SilentlyContinue
    }
}
