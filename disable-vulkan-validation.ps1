# Quick launcher for Vulkan Configuration Tool
# Use this to disable the validation layer that causes debug output spam

$vulkanConfigPath = "C:\VulkanSDK\1.4.341.1\Bin\vkconfig.exe"

if (-not (Test-Path $vulkanConfigPath)) {
    Write-Host "ERROR: Vulkan Config tool not found at $vulkanConfigPath"
    Write-Host "Please install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows"
    exit 1
}

Write-Host "Opening Vulkan Configuration Tool..."
Write-Host ""
Write-Host "To disable validation layer output:"
Write-Host "  1. Find 'Khronos Vulkan Validation Layer' in the left panel"
Write-Host "  2. Click the X button to DISABLE it, or set to 'Errors Only'"
Write-Host "  3. Click 'OK' to apply"
Write-Host ""
Write-Host "Then restart PowerShell and run your llama-vulkan commands again."
Write-Host ""

& $vulkanConfigPath
