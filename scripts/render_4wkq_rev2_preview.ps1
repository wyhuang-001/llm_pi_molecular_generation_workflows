$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$stem = '科研汇报_4WKQ真实结构_Mermaid简洁版_修订版'
$inputPath = Join-Path $root "deliverables\$stem.pptx"
$preview = Join-Path $root 'deliverables\preview_4wkq_rev2'
$pdf = Join-Path $root "deliverables\$stem.pdf"
New-Item -ItemType Directory -Force -Path $preview | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $app.Presentations.Open($inputPath, -1, 0, 0)
    $presentation.Export($preview, 'PNG', 1920, 1080)
    $presentation.SaveAs($pdf, 32)
    $checks = @()
    foreach ($slide in $presentation.Slides) {
        foreach ($sh in $slide.Shapes) {
            if ($sh.HasTextFrame -eq -1 -and $sh.TextFrame.HasText -eq -1) {
                $range = $sh.TextFrame2.TextRange
                if (($range.BoundHeight -gt ($sh.Height + 2)) -or ($range.BoundWidth -gt ($sh.Width + 2))) {
                    $checks += [pscustomobject]@{ Slide = $slide.SlideIndex; Name = $sh.Name; Text = $range.Text; Height = $sh.Height; BoundHeight = $range.BoundHeight; Width = $sh.Width; BoundWidth = $range.BoundWidth }
                }
            }
        }
    }
    ConvertTo-Json -InputObject @($checks) -Depth 4 | Set-Content -Path (Join-Path $root 'deliverables\4wkq_rev2_text_overflow.json') -Encoding UTF8
    Write-Output "Exported $($presentation.Slides.Count) slides and PDF. Text-overflow warnings: $($checks.Count)"
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
}
