# 読み上げ文（UTF-8 のテキスト）を、Windows 標準の音声合成（System.Speech）で WAV にする（2026-11-16。build_poster4_talk_audio.py から呼ぶ）。
# PC の中だけで動く（文章を外部に送らない）。
# テキストの書き方：1 行に 1 つの文のまとまり。「[間 秒数]」だけの行は、その秒数の無音。
# 実行：pwsh -NoProfile -File 04_補助スクリプト/poster4_tts.ps1 -TextFile <入力> -OutFile <出力.wav> -Voice "Microsoft Ichiro" -Rate 0
#   Rate：話す速さ（-10〜10。0 がふつう、負の数でゆっくり）。OutFile がすでにあれば止める（上書きしない）。
param(
    [Parameter(Mandatory = $true)][string]$TextFile,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [string]$Voice = "Microsoft Ichiro",
    [int]$Rate = 0
)
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutFile) {
    throw "すでにある：$OutFile（上書きしない）"
}
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
$synth.Rate = $Rate
$culture = [System.Globalization.CultureInfo]::GetCultureInfo("ja-JP")
$pb = New-Object System.Speech.Synthesis.PromptBuilder -ArgumentList $culture
$lines = [System.IO.File]::ReadAllLines($TextFile, [System.Text.Encoding]::UTF8)
foreach ($line in $lines) {
    $t = $line.Trim()
    if ($t -match '^\[間 ?([0-9.]+)\]$') {
        $pb.AppendBreak([TimeSpan]::FromSeconds([double]$Matches[1]))
    }
    elseif ($t -ne '') {
        $pb.AppendText($t)
    }
}
$synth.SetOutputToWaveFile($OutFile)
$synth.Speak($pb)
$synth.SetOutputToNull()
$synth.Dispose()
Write-Output "保存：$OutFile（声 $Voice、速さ $Rate）"
