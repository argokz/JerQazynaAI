param([string]$MigrationDate = '2026-10-21', [string]$ResumeVerifiedRecord)

$ErrorActionPreference = 'Stop'
$projectStorageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($projectStorageRoot.TrimEnd('\') -ne 'H:\projects\JerQazynaAI') { throw 'Unexpected workspace.' }
$profileStorageRoot = [IO.Path]::GetFullPath([Environment]::GetFolderPath('UserProfile'))
if ($profileStorageRoot.TrimEnd('\') -ne 'C:\Users\Danko') { throw 'Unexpected user profile.' }
$cacheStorageRoot = 'H:\caches\JerQazynaAI'
$recordDirectory = Join-Path $projectStorageRoot 'data\storage'
New-Item -ItemType Directory -Path $recordDirectory -Force | Out-Null
$recordPath = Join-Path $recordDirectory ("migration_{0}_{1}.json" -f $MigrationDate, (Get-Date -Format 'HHmmss'))
$migration = [ordered]@{
    state = 'in_progress'
    started_utc = [DateTime]::UtcNow.ToString('o')
    project = $projectStorageRoot
    cache_root = $cacheStorageRoot
    entries = [Collections.Generic.List[object]]::new()
    previous_user_environment = [ordered]@{}
}
foreach ($key in @('TEMP','TMP','TMPDIR','PIP_CACHE_DIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX','PROJ_USER_WRITABLE_DIRECTORY')) {
    $migration.previous_user_environment[$key] = [Environment]::GetEnvironmentVariable($key, 'User')
}

function Save-MigrationRecord {
    $text = $migration | ConvertTo-Json -Depth 14
    [IO.File]::WriteAllText($recordPath, $text, [Text.UTF8Encoding]::new($false))
}

function Assert-Contained([string]$Path, [string]$Parent) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $prefix = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escaped its explicitly allowed parent: $full"
    }
    return $full
}

function Assert-NoActivePythonJobs {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(-m pytest|pytest\.exe|pip install|pip\.exe)'
    })
    if ($active.Count) { throw 'A pip/pytest job is still running. Original files have been preserved.' }
}

function Move-VerifiedCache([string]$Source, [string]$Target) {
    $sourceFull = Assert-Contained $Source $profileStorageRoot
    $targetParent = if ($Target.StartsWith($cacheStorageRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $cacheStorageRoot
    } else { Join-Path $projectStorageRoot 'outputs' }
    $targetFull = Assert-Contained $Target $targetParent
    if ([IO.Path]::GetPathRoot($targetFull) -ne 'H:\') { throw 'Destination is not on H:.' }
    if (-not (Test-Path -LiteralPath $sourceFull)) { return }
    $sourceItem = Get-Item -LiteralPath $sourceFull -Force
    if (-not $sourceItem.PSIsContainer) { throw "Expected directory: $sourceFull" }
    if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        if ([IO.Path]::GetFullPath($sourceItem.Target).TrimEnd('\') -ne $targetFull) {
            throw "Unexpected existing junction at $sourceFull"
        }
        $migration.entries.Add([ordered]@{source=$sourceFull;target=$targetFull;state='already_on_h'})
        Save-MigrationRecord
        return
    }
    # Never traverse other mounts, junctions or symlinks during a recursive move.
    $links = @(Get-ChildItem -LiteralPath $sourceFull -Force -Recurse -Attributes ReparsePoint)
    if ($links.Count) { throw "Nested reparse point found: $sourceFull" }
    $entry = [ordered]@{source=$sourceFull;target=$targetFull;state='copying';bytes=0;files=@()}
    $migration.entries.Add($entry)
    Save-MigrationRecord
    if (Test-Path -LiteralPath $targetFull) {
        if (@(Get-ChildItem -LiteralPath $targetFull -Force).Count) {
            throw "Refusing to merge into a nonempty destination: $targetFull"
        }
        # Copy children when the destination was created by environment setup.
        Get-ChildItem -LiteralPath $sourceFull -Force | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $targetFull -Recurse -Force
        }
    } else {
        New-Item -ItemType Directory -Path (Split-Path $targetFull) -Force | Out-Null
        Copy-Item -LiteralPath $sourceFull -Destination $targetFull -Recurse -Force
    }
    $files = @(Get-ChildItem -LiteralPath $sourceFull -File -Force -Recurse)
    $verified = [Collections.Generic.List[object]]::new()
    $index = 0
    foreach ($file in $files) {
        $relative = [IO.Path]::GetRelativePath($sourceFull, $file.FullName)
        $copied = Join-Path $targetFull $relative
        $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not (Test-Path -LiteralPath $copied) -or
            (Get-Item -LiteralPath $copied).Length -ne $file.Length -or
            (Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash) {
            throw "Copy verification failed: $relative"
        }
        $verified.Add([ordered]@{path=$relative;bytes=$file.Length;sha256=$sourceHash})
        $entry.bytes += $file.Length
        $index++
        if ($index % 1000 -eq 0) { Write-Output "Verified $index/$($files.Count) files: $sourceFull" }
    }
    if (@(Get-ChildItem -LiteralPath $targetFull -File -Force -Recurse).Count -ne $files.Count) {
        throw 'Destination file count differs.'
    }
    $entry.files = $verified.ToArray()
    $entry.state = 'verified_copy'
    Save-MigrationRecord
    Assert-NoActivePythonJobs
    $backup = Assert-Contained ($sourceFull + '.jerqazyna-verified-move') $profileStorageRoot
    if (Test-Path -LiteralPath $backup) { throw "Backup already exists: $backup" }
    # Freeze the verified original with a same-volume rename. Use direct H: paths:
    # cross-volume reparse points are unsupported by this environment.
    Move-Item -LiteralPath $sourceFull -Destination $backup
    # Full source/target hashes have already matched. Check the final inventory
    # before removing only that verified duplicate.
    if (@(Get-ChildItem -LiteralPath $backup -File -Force -Recurse).Count -ne $verified.Count) {
        throw "Source changed during migration; preserved at $backup"
    }
    foreach ($item in $verified) {
        if ((Get-Item -LiteralPath (Join-Path $backup $item.path)).Length -ne $item.bytes) {
            throw "Source changed during migration; preserved at $backup"
        }
    }
    $backup = Assert-Contained $backup $profileStorageRoot
    if (-not $backup.EndsWith('.jerqazyna-verified-move')) { throw 'Unsafe backup removal path.' }
    if ((Get-Item -LiteralPath $backup -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Backup unexpectedly became a junction.'
    }
    Remove-Item -LiteralPath $backup -Recurse -Force
    $entry.state = 'moved_verified_direct_h'
    Save-MigrationRecord
    Write-Output "Moved $($verified.Count) files / $($entry.bytes) bytes to $targetFull"
}

try {
    Assert-NoActivePythonJobs
    Save-MigrationRecord
    if ($ResumeVerifiedRecord) {
        $oldPath = Assert-Contained $ResumeVerifiedRecord $recordDirectory
        $old = Get-Content -LiteralPath $oldPath -Raw | ConvertFrom-Json
        $previous = @($old.entries | Where-Object { $_.source -eq "$profileStorageRoot\AppData\Local\pip\cache" })[0]
        if ($previous.state -ne 'verified_copy' -or $previous.target -ne "$cacheStorageRoot\pip" -or $previous.files.Count -ne 6689) {
            throw 'Unexpected previously verified cache record.'
        }
        $cachedOriginal = Assert-Contained "$profileStorageRoot\AppData\Local\pip-before-h-storage-2026-10-21\cache.jerqazyna-verified-move" $profileStorageRoot
        if ((Get-Item -LiteralPath $cachedOriginal -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected original type.' }
        if (@(Get-ChildItem -LiteralPath $cachedOriginal -Force -Recurse -Attributes ReparsePoint).Count) { throw 'Unexpected nested link.' }
        if (@(Get-ChildItem -LiteralPath $cachedOriginal -File -Force -Recurse).Count -ne $previous.files.Count) { throw 'Original changed.' }
        foreach ($file in $previous.files) {
            $originalFile = Assert-Contained (Join-Path $cachedOriginal $file.path) $cachedOriginal
            $targetFile = Assert-Contained (Join-Path "$cacheStorageRoot\pip" $file.path) "$cacheStorageRoot\pip"
            if ((Get-Item -LiteralPath $originalFile).Length -ne $file.bytes -or (Get-Item -LiteralPath $targetFile).Length -ne $file.bytes) { throw 'Verified inventory changed.' }
        }
        $recovered = [ordered]@{source=$cachedOriginal;target=$previous.target;state='verified_copy_recovered';bytes=$previous.bytes;files=$previous.files;prior_verification_record=$oldPath}
        $migration.entries.Add($recovered)
        Save-MigrationRecord
        Assert-NoActivePythonJobs
        $cachedOriginal = Assert-Contained $cachedOriginal $profileStorageRoot
        if (-not $cachedOriginal.EndsWith('.jerqazyna-verified-move')) { throw 'Unsafe removal path.' }
        Remove-Item -LiteralPath $cachedOriginal -Recurse -Force
        $recovered.state = 'moved_verified_direct_h'
        $migration.retired_empty_reparse_entry = "$profileStorageRoot\AppData\Local\pip-before-h-storage-2026-10-21\cache"
        $migration.retired_entry_file_bytes = 0
        Save-MigrationRecord
        Write-Output 'Committed the fully SHA-256-verified pip copy on H: and removed its original duplicate.'
    } else {
        Move-VerifiedCache "$profileStorageRoot\AppData\Local\pip\cache" "$cacheStorageRoot\pip"
    }
    Move-VerifiedCache "$profileStorageRoot\.matplotlib" "$cacheStorageRoot\matplotlib"
    Move-VerifiedCache "$profileStorageRoot\AppData\Local\proj" "$cacheStorageRoot\proj"
    Move-VerifiedCache "$profileStorageRoot\AppData\Local\Temp\pytest-of-Danko" "$cacheStorageRoot\history\pytest-of-Danko"
    $oldPipTemp = @(Get-ChildItem -LiteralPath "$profileStorageRoot\AppData\Local\Temp" -Directory -Force |
        Where-Object { $_.Name -like 'pip-*' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })
    foreach ($item in $oldPipTemp) {
        Move-VerifiedCache $item.FullName (Join-Path "$cacheStorageRoot\history\pip-temp" $item.Name)
    }
    Move-VerifiedCache "$profileStorageRoot\.codex\visualizations\2026\10\19\01a1523f-6462-7443-8d74-e7192d9f5178" `
        "$projectStorageRoot\outputs\visualizations\01a1523f-6462-7443-8d74-e7192d9f5178"
    & (Join-Path $PSScriptRoot 'use-h-storage.ps1') -Persist
    $migration.state = 'completed'
    $migration.completed_utc = [DateTime]::UtcNow.ToString('o')
    $migration.moved_bytes = [long]0
    foreach ($movedEntry in $migration.entries) {
        $migration.moved_bytes += [long]$movedEntry['bytes']
    }
    Save-MigrationRecord
    Write-Output "Migration record: $recordPath"
} catch {
    $migration.state = 'incomplete_originals_preserved'
    $migration.error_type = $_.Exception.GetType().Name
    Save-MigrationRecord
    throw
}
