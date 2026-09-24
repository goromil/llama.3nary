param([Parameter(ValueFromRemainingArguments=$true)]$Args)

# Set absolute path to nv binaries
$llamaBin = "$env:USERPROFILE\.local\3nary-nv"

if (-not ($env:Path -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $llamaBin })) {
    $env:PATH = "$llamaBin;$env:PATH"
}

# Extract executable name (first argument) and remaining arguments
if ($Args.Count -eq 0) {
    Write-Host "Usage: .\run-3nary-nv.ps1 <executable-name> [arguments...]"
    Write-Host "Example: .\run-3nary-nv.ps1 cli -m model.gguf"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem "$llamaBin\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
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

# Run the selected executable with remaining arguments
& $executablePath @remainingArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "ERROR: executable exited with code $exitCode"
}
exit $exitCode
