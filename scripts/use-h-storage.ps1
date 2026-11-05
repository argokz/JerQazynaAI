param([switch]$Persist)

$ErrorActionPreference = 'Stop'
$projectStorageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([IO.Path]::GetPathRoot($projectStorageRoot) -ne 'H:\') {
    throw 'JerQazynaAI storage must be on H:.'
}
$cacheStorageRoot = 'H:\caches\JerQazynaAI'
$projectStorageEnvironment = [ordered]@{
    TEMP = "$cacheStorageRoot\tmp"
    TMP = "$cacheStorageRoot\tmp"
    TMPDIR = "$cacheStorageRoot\tmp"
    PIP_CACHE_DIR = "$cacheStorageRoot\pip"
    MPLCONFIGDIR = "$cacheStorageRoot\matplotlib"
    PYTHONPYCACHEPREFIX = "$cacheStorageRoot\pycache"
    PROJ_USER_WRITABLE_DIRECTORY = "$cacheStorageRoot\proj"
    XDG_CACHE_HOME = "$cacheStorageRoot\xdg"
}
foreach ($entry in $projectStorageEnvironment.GetEnumerator()) {
    New-Item -ItemType Directory -Path $entry.Value -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    if ($Persist -and $entry.Key -ne 'XDG_CACHE_HOME' -and
        [Environment]::GetEnvironmentVariable($entry.Key, 'User') -ne $entry.Value) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'User')
    }
}
# Apply before application imports even when a running desktop app has inherited
# old C: environment variables. No packages or baseline artifacts are modified.
$pythonStorageEnvironment = [ordered]@{}
foreach ($entry in $projectStorageEnvironment.GetEnumerator()) {
    $pythonStorageEnvironment[$entry.Key] = $entry.Value.Replace('\', '/')
}
$storageJson = $pythonStorageEnvironment | ConvertTo-Json -Compress
$startupLine = "import sys,os; sys.pycache_prefix='$($pythonStorageEnvironment.PYTHONPYCACHEPREFIX)'; os.environ.update($storageJson)"
foreach ($venvName in @('.venv', '.venv-emit')) {
    $sitePackages = Join-Path $projectStorageRoot "$venvName\Lib\site-packages"
    if (Test-Path -LiteralPath $sitePackages) {
        $startupFile = Join-Path $sitePackages 'zz_jerqazyna_h_storage.pth'
        $startupText = "# JerQazynaAI: project caches and temporary files on H:`n$startupLine`n"
        if ((Test-Path -LiteralPath $startupFile) -and
            (Get-Content -LiteralPath $startupFile -Raw) -notlike '# JerQazynaAI:*') {
            throw "Existing startup file is not managed by this project: $startupFile"
        }
        [IO.File]::WriteAllText($startupFile, $startupText, [Text.UTF8Encoding]::new($false))
    }
}
Write-Output "Project data: $projectStorageRoot"
Write-Output "Cache and temporary files: $cacheStorageRoot"
