param(
    [Parameter(Mandatory=$true, Position=0)]
    $script,

    [Parameter(Mandatory=$false, Position=1)]
    [ValidateRange(1, 16)]
    [int]$threads = 1,

    [Parameter(Mandatory=$false, Position=2)]
    $fptlexe = ".fptl/fptl.exe"
)


#Read more: https://www.sharepointdiary.com/2021/02/powershell-function-parameters.html#ixzz8Vu58LpEi

Write-Host "RUN fptl with ${threads} threads:" -ForegroundColor Cyan

$t0 = Get-Date
& $fptlexe $script -n $threads
$t1 = Get-Date
$dt = New-TimeSpan -Start $t0 -End $t1

Write-Host "DONE in $dt" -ForegroundColor Cyan
Write-Host