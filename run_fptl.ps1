param(
    [Parameter(Mandatory=$true, Position=0)]
    $script,
    [Parameter(Mandatory=$false, Position=2)]
    $fptlexe = ".fptl/fptl.exe"
)


for ($threads = 1; $threads -le 8; $threads = $threads + 1){
    & $fptlexe $script -n $threads
    Write-Host `t -NoNewline
}
