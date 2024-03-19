param(
    [Parameter(Mandatory)]$version,
    [Parameter()]$dbg

)

Write-Host "RUN fptl_$version : " -ForegroundColor Cyan

$t0 = Get-Date
& ".fptl_$version\fptl.exe" "c:\Users\istrof10\source\repos\FoxAlgorithm\fox_fptl.fptl"
$t1 = Get-Date
$dt = New-TimeSpan -Start $t0 -End $t1

Write-Host "FPTL DONE in $dt" -ForegroundColor Cyan
Write-Host