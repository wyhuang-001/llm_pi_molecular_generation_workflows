param(
    [Parameter(Mandatory=$true)][string]$Stem,
    [string]$PreviewName = 'preview_4wkq'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$inputPath = Join-Path $root "deliverables\$Stem.pptx"
$preview = Join-Path $root "deliverables\$PreviewName"
$pdf = Join-Path $root "deliverables\$Stem.pdf"
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
                if (($range.BoundHeight -gt ($shape.Height + 2)) -or ($range.BoundWidth -gt ($shape.Width + 2))) {
                    $checks += [pscustomobject]@{
                        Slide = $slide.SlideIndex
                        Text = $range.Text
                        Height = $shape.Height
                        BoundHeight = $range.BoundHeight
                        Width = $shape.Width
                        BoundWidth = $range.BoundWidth
                    }
                }
            }
        }
    }
    ConvertTo-Json -InputObject @($checks) -Depth 4 | Set-Content -Path (Join-Path $preview 'text_overflow.json') -Encoding UTF8
    Write-Output "Exported $($presentation.Slides.Count) slides and PDF. Text-overflow warnings: $($checks.Count)"
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    # Keep the user's other open presentations intact.
}
