param([Parameter(ValueFromRemainingArguments=$true)]$Args)

# Setup Vulkan environment
$vulkanSdkPath = "C:\VulkanSDK\1.4.341.1"

if (-not (Test-Path $vulkanSdkPath)) {
    Write-Host "ERROR: Vulkan SDK not found at $vulkanSdkPath"
    Write-Host "Please install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows"
    exit 1
}

$env:LLAMA_PATH = "$env:USERPROFILE\.local\3nary-vulkan"

# Add Vulkan to environment
$env:VULKAN_SDK = $vulkanSdkPath
$env:PATH = "$vulkanSdkPath\Bin;$env:LLAMA_PATH;$env:PATH"

# Disable Vulkan validation layer verbose output
$env:VK_INSTANCE_LAYERS = ""
$env:VK_LAYER_LUNARG_standard_validation = "0"
$env:VK_LAYER_KHRONOS_validation = "0"
$env:VK_LAYER_PATH = ""
# Suppress detailed layer output
$env:VK_LOADER_DEBUG = "error"
$env:VK_LOADER_LAYERS_ENABLE = ""

# Extract executable name (first argument) and remaining arguments
if ($Args.Count -eq 0) {
    Write-Host "Usage: .\run-3nary-vulkan.ps1 <executable-name> [arguments...]"
    Write-Host "Example: .\run-3nary-vulkan.ps1 cli -m model.gguf -ngl 99"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem "$env:LLAMA_PATH\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    exit 0
}

$exeName = $Args[0]
$executablePath = "$env:LLAMA_PATH\llama-$exeName.exe"
$remainingArgs = $Args[1..($Args.Count-1)]

# Verify executable exists
if (-not (Test-Path $executablePath)) {
    Write-Host "ERROR: Executable not found: $executablePath"
    Write-Host ""
    Write-Host "Available executables:"
    Get-ChildItem "$env:LLAMA_PATH\llama-*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace "llama-", ""
        Write-Host "  - $name"
    }
    exit 1
}

Write-Host "Vulkan SDK configured: $vulkanSdkPath"
Write-Host "Running $exeName with Vulkan GPU support..."
Write-Host ""

# Suppress validation layer output (requires system-level config via vkconfig)
$env:VK_INSTANCE_LAYERS = ""
$env:VK_LAYER_KHRONOS_validation = "0"
$env:VK_LOADER_DEBUG = "error"

# Run the selected executable with remaining arguments
& $executablePath @remainingArgs
