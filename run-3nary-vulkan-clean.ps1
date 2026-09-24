# Workaround script: Filter out Vulkan validation layer debug output
# Usage: ./run-3nary-vulkan-clean.ps1 server -m model.gguf -ngl 99

param([Parameter(ValueFromRemainingArguments=$true)]$Args)

# Setup Vulkan environment
$vulkanSdkPath = "C:\VulkanSDK\1.4.341.1"

if (-not (Test-Path $vulkanSdkPath)) {
    Write-Host "ERROR: Vulkan SDK not found at $vulkanSdkPath"
    Write-Host "Please install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows"
    exit 1
}

# Add Vulkan to environment
$env:VULKAN_SDK = $vulkanSdkPath
$env:PATH = "$vulkanSdkPath\Bin;$env:PATH"

# Disable Vulkan validation layer output
$env:VK_INSTANCE_LAYERS = ""
$env:VK_LAYER_KHRONOS_validation = "0"
$env:VK_LOADER_DEBUG = "error"

# Check if executable name provided
if ($Args.Count -eq 0) {
    Write-Host "Usage: .\run-3nary-vulkan-clean.ps1 <executable-name> [arguments...]"
    Write-Host "Example: .\run-3nary-vulkan-clean.ps1 server -m model.gguf -ngl 99"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem ".\build\bin\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    Write-Host ""
    Write-Host "NOTE: If you still see Vulkan debug output, disable it globally via:"
    Write-Host "  .\disable-vulkan-validation.ps1"
    exit 0
}

$exeName = $Args[0]
$executablePath = ".\build\bin\llama-$exeName.exe"
$remainingArgs = $Args[1..($Args.Count-1)]

# Verify executable exists
if (-not (Test-Path $executablePath)) {
    Write-Host "ERROR: Executable not found: $executablePath"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem ".\build\bin\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    exit 1
}

Write-Host "Vulkan SDK configured: $vulkanSdkPath"
Write-Host "Running $exeName with Vulkan GPU support..."
Write-Host "(Filtering Vulkan validation layer output)"
Write-Host ""

# Run the executable with filtering
# This filters out Vulkan validation layer tracing output
& $executablePath @remainingArgs 2>&1 | Where-Object {
    # Keep non-empty lines that are NOT Vulkan validation tracing
    $_ -ne "" -and `
    $_ -notmatch "^Thread \d+, Frame \d+:" -and `
    $_ -notmatch "^.*returns VkResult" -and `
    $_ -notmatch "^.*returns void:" -and `
    $_ -notmatch "^\s{20,}" -and `
    $_ -notmatch "pCreateInfo|pAllocator|pInstance|pNext|sType|layout|device" -and `
    $_ -notmatch "buffer|memory|fence|pipeline|shader" -and `
    $_ -notmatch "^$"
}
