Write-Host ">>> MASTER OPERATOR BOOTSTRAP SEQUENCE INITIATED" -ForegroundColor Cyan

# 1. Clear the socket port to ensure a clean binding
./clear_port.ps1 -Port 65432

# 2. Start the Agentic Kernel
Write-Host ">>> Launching Agentic Kernel..." -ForegroundColor Green
$pythonPath = "c:\Users\eqhsp\AppData\Local\Programs\Python\Python313\python.exe"
& $pythonPath kernel.py
