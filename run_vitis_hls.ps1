$vitisRoot = 'E:\2025.2\Vitis'
$env:HLS_PART = 'xc7z020clg400-1'
$env:HLS_CLOCK_PERIOD = '10.0'
$env:HLS_FLOW = if ($args.Length -ge 1) { $args[0] } else { 'all' }
$env:XILINX_VITIS = $vitisRoot
$env:PYTHONPATH = "$vitisRoot\cli;$vitisRoot\cli\python-packages\win64;$vitisRoot\cli\python-packages\site-packages;$vitisRoot\cli\proto;$vitisRoot\scripts\python_pkg"
$env:PATH = "$vitisRoot\bin;$env:PATH"

$cliPath = Join-Path $PSScriptRoot 'run_hls_cli.py'
python $cliPath
