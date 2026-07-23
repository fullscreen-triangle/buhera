# purpose-reindex.ps1 — rebuild the Purpose (Tool B) index, then strip vendored /
# build-output symbols that `purpose index` does not skip on its own.
#
# See purpose-reindex.sh for the full rationale. Run this on Windows/PowerShell:
#   .\scripts\purpose-reindex.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$Index     = Join-Path $RepoRoot ".purpose\index.json"

Write-Host "[purpose-reindex] indexing $RepoRoot ..."
& purpose index --root $RepoRoot

if (-not (Test-Path $Index)) {
    Write-Error "[purpose-reindex] $Index not produced by 'purpose index'."
    exit 1
}

Write-Host "[purpose-reindex] filtering vendored / build-output symbols ..."
& python (Join-Path $ScriptDir "purpose_filter_index.py") $Index

Write-Host "[purpose-reindex] done. Ask with:  purpose ask `"<question>`""
