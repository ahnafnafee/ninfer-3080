# ninfer-switch <profile>|start|restart|status|kill|logs [-f]|list
#
# Windows counterpart of a systemd-driven switcher: there is no service manager here, so the server
# runs as a hidden background process that outlives the terminal, and the choice is remembered in a
# state file so `start` and `restart` bring back the last profile.
#
# Profiles come from profiles.json next to this script. NINFER_PROFILES names a second file whose
# profiles are added on top (same name replaces), and whose "models" and "port" override the
# defaults. In exe and args, {repo} is this checkout, {models} is the models directory
# (NINFER_MODELS, else the overlay's "models", else {repo}\models), and {here} is the directory of
# the file that defines the profile. --host and --port are appended, so profiles never repeat them.
param([Parameter(Position = 0)][string]$Command = 'help', [Parameter(Position = 1)][string]$Arg)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$StateDir = Join-Path $env:LOCALAPPDATA 'ninfer'
$StateFile = Join-Path $StateDir 'switch-state.json'
$Log = Join-Path $StateDir 'serve.log'

function Read-Profiles {
    $files = @(Join-Path $PSScriptRoot 'profiles.json')
    if ($env:NINFER_PROFILES) { $files += $env:NINFER_PROFILES }
    $cfg = @{ port = 8080; models = (Join-Path $Repo 'models'); profiles = [ordered]@{} }
    foreach ($f in $files) {
        if (-not (Test-Path -LiteralPath $f)) { throw "profiles file not found: $f" }
        $j = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json
        if ($j.port) { $cfg.port = [int]$j.port }
        if ($j.models) { $cfg.models = $j.models }
        foreach ($p in $j.profiles.PSObject.Properties) {
            $p.Value | Add-Member -NotePropertyName here -NotePropertyValue (Split-Path -Parent (Resolve-Path -LiteralPath $f)) -Force
            $cfg.profiles[$p.Name] = $p.Value
        }
    }
    if ($env:NINFER_MODELS) { $cfg.models = $env:NINFER_MODELS }
    if ($env:NINFER_PORT) { $cfg.port = [int]$env:NINFER_PORT }
    $cfg
}

function Expand([string]$s, $cfg, $p) {
    $s.Replace('{repo}', $Repo).Replace('{models}', $cfg.models).Replace('{here}', $p.here)
}

function Read-State { if (Test-Path -LiteralPath $StateFile) { Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json } }

function Get-ServerNames($cfg) {
    @($cfg.profiles.Values | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_.exe) }) | Select-Object -Unique
}

# Another app can hold the same port on another address (a dual-stack ::), so a listener only
# counts as ours when a server binary this tool starts owns it. Returns ours first, else a foreign one.
function Get-Owner($cfg) {
    $owners = @(Get-NetTCPConnection -LocalPort $cfg.port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue })
    $mine = $owners | Where-Object { $_.ProcessName -in (Get-ServerNames $cfg) } | Select-Object -First 1
    if ($mine) { [pscustomobject]@{ Process = $mine; Ours = $true } }
    elseif ($owners) { [pscustomobject]@{ Process = $owners[0]; Ours = $false } }
}

function Test-Ready($cfg) {
    $o = Get-Owner $cfg
    if (-not ($o -and $o.Ours)) { return $false }
    try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$($cfg.port)/health").StatusCode -eq 200 } catch { $false }
}

function Stop-Server($cfg) {
    $state = Read-State
    $ids = @()
    if ($state -and (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)) { $ids += $state.pid }
    $o = Get-Owner $cfg
    if ($o -and $o.Ours) { $ids += $o.Process.Id }
    if (-not $ids) { return $false }
    # The launcher is a cmd.exe wrapping the server, so end the whole tree.
    foreach ($id in ($ids | Select-Object -Unique)) { taskkill /PID $id /T /F *> $null }
    for ($i = 0; $i -lt 30 -and (Get-Owner $cfg).Ours; $i++) { Start-Sleep 1 }
    $true
}

function Start-Server([string]$name, $cfg) {
    $p = $cfg.profiles[$name]
    if (-not $p) { throw "unknown profile '$name'. Run: ninfer-switch list" }
    $exe = Expand $p.exe $cfg $p
    $argv = @($p.args | ForEach-Object { Expand $_ $cfg $p }) + @('--host', '127.0.0.1', '--port', "$($cfg.port)")
    if (-not (Test-Path -LiteralPath $exe)) { throw "server binary not found: $exe" }
    foreach ($a in $argv) {
        if ($a -match '\.(ninfer|gguf)$' -and -not (Test-Path -LiteralPath $a)) { throw "model not found: $a" }
    }
    if (Stop-Server $cfg) { Write-Host 'stopped the running server' }
    $o = Get-Owner $cfg
    if ($o -and -not $o.Ours) {
        throw "port $($cfg.port) is taken by $($o.Process.ProcessName) (pid $($o.Process.Id)); set NINFER_PORT or ""port"" in profiles"
    }
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $quoted = (@($exe) + $argv | ForEach-Object { if ($_ -match '[\s"&|<>^]') { '"' + $_ + '"' } else { $_ } }) -join ' '
    # cmd /s strips exactly the outer quotes, so the inner ones survive and both streams share one log.
    $proc = Start-Process -FilePath $env:ComSpec -ArgumentList "/d /s /c `"$quoted > `"$Log`" 2>&1`"" -WindowStyle Hidden -PassThru
    [ordered]@{ profile = $name; pid = $proc.Id; started = (Get-Date).ToString('o'); port = $cfg.port } |
        ConvertTo-Json | Set-Content -LiteralPath $StateFile -Encoding utf8
    Write-Host "starting $name ($($p.note))"
    for ($i = 0; $i -lt 300; $i++) {
        if (Test-Ready $cfg) { Write-Host "serving $name on http://127.0.0.1:$($cfg.port)/v1"; return }
        if ($proc.HasExited) { break }
        if ($i % 5 -eq 4 -and (Test-Path -LiteralPath $Log)) {
            $last = Get-Content -LiteralPath $Log -Tail 1
            if ($last) { Write-Host ('  ' + $last.Substring(0, [math]::Min(110, $last.Length))) }
        }
        Start-Sleep 1
    }
    Write-Host "failed to start $name; last log lines:" -ForegroundColor Red
    if (Test-Path -LiteralPath $Log) { Get-Content -LiteralPath $Log -Tail 8 | ForEach-Object { "  $_" } }
    exit 1
}

function Show-Status($cfg) {
    $state = Read-State
    $proc = if ($state) { Get-Process -Id $state.pid -ErrorAction SilentlyContinue }
    $o = Get-Owner $cfg
    $ours = $o -and $o.Ours
    if ($proc -or $ours) {
        # While weights load nothing listens yet, so fall back to the launcher process.
        $server = if ($ours) { $o.Process } else { $proc }
        $since = if ($state) { [datetime]$state.started } else { $server.StartTime }
        $up = (Get-Date) - $since
        $name = if ($state) { $state.profile } else { 'unknown (not started by ninfer-switch)' }
        $note = if ($state -and $cfg.profiles.Contains($state.profile)) { " ($($cfg.profiles[$state.profile].note))" }
        "profile   : $name$note"
        "process   : $($server.ProcessName) pid $($server.Id), up {0:%d}d {0:hh}h {0:mm}m (since {1:yyyy-MM-dd HH:mm})" -f $up, $since
    } else {
        "profile   : none running (last: $(if ($state) { $state.profile } else { 'none' }))"
    }
    if ($ours) {
        "port      : $($cfg.port) listening"
        $served = try { ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$($cfg.port)/v1/models").Content | ConvertFrom-Json).data[0].id } catch { 'no response' }
        "serving   : $served  (health: $(if (Test-Ready $cfg) { 'ok' } else { 'not ready' }))"
    } elseif ($o) {
        "port      : $($cfg.port) taken by $($o.Process.ProcessName) (pid $($o.Process.Id)), not a server this tool runs"
    } else {
        "port      : $($cfg.port) NOT listening"
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $g = (nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1) -split ',\s*'
        "gpu       : $($g[0]) MiB used, $($g[1]) MiB free, $($g[2]) C, $($g[3])% busy"
    }
    if (Test-Path -LiteralPath $Log) {
        $cap = Select-String -LiteralPath $Log -Pattern 'capacity \|' | Select-Object -Last 1
        if ($cap) { "capacity  : " + $cap.Line.Substring($cap.Line.IndexOf('capacity |') + 11) }
        $tput = Select-String -LiteralPath $Log -Pattern 'throughput \|' | Select-Object -Last 1
        if ($tput) { "last rate : " + $tput.Line.Substring($tput.Line.IndexOf('throughput |') + 13) }
        $errs = @(Select-String -LiteralPath $Log -Pattern '\b(ERROR|FATAL)\b|\berror:' ).Count
        "log       : $Log ($errs errors)"
    }
}

$cfg = Read-Profiles
switch ($Command) {
    { $_ -in 'status', 'st' } { Show-Status $cfg; break }
    { $_ -in 'kill', 'stop' } {
        if (Stop-Server $cfg) { 'stopped' } else { 'nothing running' }
        break
    }
    { $_ -in 'start', 'restart' } {
        $state = Read-State
        $name = if ($Arg) { $Arg } elseif ($state) { $state.profile } else { @($cfg.profiles.Keys)[0] }
        Start-Server $name $cfg
        break
    }
    'logs' {
        if (-not (Test-Path -LiteralPath $Log)) { 'no log yet'; break }
        if ($Arg -eq '-f') { Get-Content -LiteralPath $Log -Tail 30 -Wait } else { Get-Content -LiteralPath $Log -Tail 40 }
        break
    }
    { $_ -in 'help', 'list', '-h', '--help' } {
        'usage: ninfer-switch <profile>|start [profile]|restart|status|kill|logs [-f]|list'
        ''
        'profiles:'
        foreach ($k in $cfg.profiles.Keys) { '  {0,-8} {1}' -f $k, $cfg.profiles[$k].note }
        break
    }
    default { Start-Server $Command $cfg }
}
