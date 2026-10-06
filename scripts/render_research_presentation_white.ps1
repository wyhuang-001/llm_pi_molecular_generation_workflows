$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$stem = '科研汇报_保留骨架的分子优化智能体_白底紧凑版'
$inputPath = Join-Path $root "deliverables\$stem.pptx"
$preview = Join-Path $root 'deliverables\preview_white'
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
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $range = $shape.TextFrame2.TextRange
                $bh = $range.BoundHeight
                $bw = $range.BoundWidth
                if (($bh -gt ($shape.Height + 2)) -or ($bw -gt ($shape.Width + 2))) {
                    $checks += [pscustomobject]@{
                        Slide = $slide.SlideIndex
                        Name = $shape.Name
                        Text = $range.Text
                        Height = $shape.Height
                        BoundHeight = $bh
                        Width = $shape.Width
                        BoundWidth = $bw
                    }
                }
            }
        }
    }
    $checkPath = Join-Path $root 'deliverables\white_revision_text_overflow.json'
    ConvertTo-Json -InputObject @($checks) -Depth 4 | Set-Content -Path $checkPath -Encoding UTF8
    Write-Output "Exported $($presentation.Slides.Count) slides and PDF. Text-overflow warnings: $($checks.Count)"
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    # Do not quit PowerPoint: the user may have other presentations open.
}
