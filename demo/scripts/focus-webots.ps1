Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ArtemisCaptureWindow {
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}
'@
$captureProcess = Get-Process -Name webots-bin -ErrorAction Stop | Where-Object MainWindowHandle | Select-Object -First 1
if (-not $captureProcess) { throw 'No visible Webots window to record.' }
[ArtemisCaptureWindow]::ShowWindow($captureProcess.MainWindowHandle, 9) | Out-Null
[ArtemisCaptureWindow]::ShowWindow($captureProcess.MainWindowHandle, 3) | Out-Null
[ArtemisCaptureWindow]::SetForegroundWindow($captureProcess.MainWindowHandle) | Out-Null
