param(
    [Parameter(Mandatory=$true)]
    [int]$Port
)

Write-Host "Checking port $Port..."

# Find any process using the port
$connection = netstat -ano | Select-String ":$Port " | Select-Object -First 1

if ($connection) {
    # Extract PID (last element in the line)
    $pidStr = ($connection -split "\s+")[-1]
    
    # Handle potential edge case where line ends with space or something
    if (-not $pidStr) {
        $pidStr = ($connection -split "\s+")[-2]
    }

    Write-Host "Port $Port is in use by PID $pidStr. Terminating..."

    try {
        Stop-Process -Id $pidStr -Force -ErrorAction Stop
        Write-Host "Process $pidStr terminated."
    }
    catch {
        Write-Host "Failed to terminate PID $pidStr. Error: $_"
    }

    Start-Sleep -Seconds 1
}
else {
    Write-Host "Port $Port is free."
}

# Double-check
$stillBound = netstat -ano | Select-String ":$Port "

if ($stillBound) {
    Write-Host "Warning: Port $Port still appears bound. OS may be in TIME_WAIT."
} else {
    Write-Host "Port $Port is now clear."
}
