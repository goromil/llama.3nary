param()

# Source Intel oneAPI environment if available
$oneApiSetvars = 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'
if (Test-Path $oneApiSetvars) {
    cmd /c """$oneApiSetvars"" intel64 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

# Set absolute path to SYCL binaries
$llamaBin = "$env:USERPROFILE\.local\3nary-sycl"

if (-not ($env:Path -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $llamaBin })) {
    $env:PATH = "$llamaBin;$env:PATH"
}

# Extract executable name (first argument) and remaining arguments
if ($Args.Count -eq 0) {
    Write-Host "Usage: .\run-3nary-sycl.ps1 <executable-name> [arguments...]"
    Write-Host "Example: .\run-3nary-sycl.ps1 cli -m model.gguf -ngl 99"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem "$llamaBin\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    Write-Host ""
    Write-Host "Common SYCL environment variables:"
    Write-Host "  GGML_SYCL_DEVICE=0           Select GPU device"
    Write-Host "  GGML_SYCL_DEBUG=1            Enable SYCL debug output"
    Write-Host "  GGML_SYCL_MEMTRACE=1         Enable memory allocation tracing"
    Write-Host "  GGML_SYCL_ENABLE_MKL_FA=1    Enable MKL flash attention"
    exit 0
}

$exeName = $Args[0]
if ($exeName.StartsWith('llama-', [System.StringComparison]::OrdinalIgnoreCase)) {
    $exeName = $exeName.Substring(6)
}
$executablePath = "$llamaBin\llama-$exeName.exe"
$remainingArgs = @()
if ($Args.Count -gt 1) {
    $remainingArgs = @($Args[1..($Args.Count-1)])
}

# Verify executable exists
if (-not (Test-Path $executablePath)) {
    Write-Host "ERROR: Executable not found: $executablePath"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem "$llamaBin\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    exit 1
}

# Run the selected executable with remaining arguments (avoid PowerShell param binding)
if ($remainingArgs.Count -gt 0) {
    $argString = ($remainingArgs | ForEach-Object { if ($_ -match ' ') { "`"$_`"" } else { $_ } }) -join ' '
    cmd /c ""$executablePath" $argString"
} else {
    & $executablePath
}
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "ERROR: executable exited with code $exitCode"
}
exit $exitCode
