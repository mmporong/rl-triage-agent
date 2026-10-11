param(
    [Parameter(Mandatory = $true)][string]$SpecPath,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$spec = Get-Content -Raw -Encoding UTF8 -LiteralPath $SpecPath | ConvertFrom-Json
$fontRegular = "C:\Windows\Fonts\malgun.ttf"
$fontBold = "C:\Windows\Fonts\malgunbd.ttf"
if (-not (Test-Path -LiteralPath $fontRegular) -or -not (Test-Path -LiteralPath $fontBold)) {
    throw "맑은 고딕 글꼴을 찾지 못했습니다."
}

$privateFonts = [System.Drawing.Text.PrivateFontCollection]::new()
$privateFonts.AddFontFile($fontRegular)
$privateFonts.AddFontFile($fontBold)
$regularFamily = $privateFonts.Families | Where-Object Name -eq "Malgun Gothic" | Select-Object -First 1
$boldFamily = $regularFamily

function New-Font([float]$Size, [System.Drawing.FontStyle]$Style) {
    return [System.Drawing.Font]::new($regularFamily, $Size, $Style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Draw-TextBlock($Graphics, [string]$Text, [float]$X, [float]$Y, [float]$Width, [float]$Height, $Font, $Brush) {
    $format = [System.Drawing.StringFormat]::new()
    $format.Trimming = [System.Drawing.StringTrimming]::EllipsisWord
    $format.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit
    $Graphics.DrawString($Text, $Font, $Brush, [System.Drawing.RectangleF]::new($X, $Y, $Width, $Height), $format)
    $format.Dispose()
}

function Draw-FittedTextBlock($Graphics, [string]$Text, [float]$X, [float]$Y, [float]$Width, [float]$Height, [float]$StartSize, [float]$MinSize, [System.Drawing.FontStyle]$Style, $Brush) {
    $format = [System.Drawing.StringFormat]::new()
    $format.Trimming = [System.Drawing.StringTrimming]::EllipsisWord
    $format.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit
    $font = $null
    $fits = $false
    for ($size = $StartSize; $size -ge $MinSize; $size -= 1) {
        if ($null -ne $font) { $font.Dispose() }
        $font = New-Font $size $Style
        $measured = $Graphics.MeasureString($Text, $font, [int]$Width, $format)
        if ($measured.Width -le ($Width + 1) -and $measured.Height -le ($Height + 1)) {
            $fits = $true
            break
        }
    }
    if (-not $fits) {
        if ($null -ne $font) { $font.Dispose() }
        $format.Dispose()
        throw "Text does not fit card bounds: $Text"
    }
    $Graphics.DrawString($Text, $font, $Brush, [System.Drawing.RectangleF]::new($X, $Y, $Width, $Height), $format)
    $font.Dispose()
    $format.Dispose()
}

foreach ($card in $spec.cards) {
    $target = Join-Path $OutputDir $card.filename
    if (Test-Path -LiteralPath $target) { throw "기존 파일을 덮어쓸 수 없습니다: $target" }

    $bitmap = [System.Drawing.Bitmap]::new(1280, 720, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    if ($card.layout -eq "reference_overlay") {
        $graphics.Clear([System.Drawing.Color]::Transparent)
    } else {
        $graphics.Clear([System.Drawing.Color]::FromArgb(250, 251, 253))
    }

    $ink = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(19, 30, 48))
    $muted = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(77, 91, 112))
    $blue = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(25, 105, 220))
    $orange = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(220, 102, 24))
    $red = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(190, 47, 47))
    $green = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(21, 132, 91))
    $white = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
    $panel = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(238, 243, 250))
    $accentBrush = switch ($card.accent) { "red" { $red } "orange" { $orange } "green" { $green } default { $blue } }

    if ($card.layout -eq "reference_overlay") {
        $top = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(238, 255, 255, 255))
        $bottom = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(238, 255, 255, 255))
        $graphics.FillRectangle($top, 0, 0, 1280, 165)
        $graphics.FillRectangle($bottom, 0, 535, 1280, 185)
        $graphics.FillRectangle($blue, 30, 172, 590, 52)
        $graphics.FillRectangle($red, 660, 172, 590, 52)
        $labelFont = New-Font 28 ([System.Drawing.FontStyle]::Bold)
        Draw-TextBlock $graphics "기존 기준 재생 · 100 iter" 50 179 550 45 $labelFont $white
        Draw-TextBlock $graphics "결함 재생 · 100 iter" 680 179 550 45 $labelFont $white
        $labelFont.Dispose()
    } else {
        $graphics.FillRectangle($accentBrush, 0, 0, 18, 720)
        $graphics.FillRectangle($panel, 62, 245, 1156, 375)
    }

    $badgeFont = New-Font 26 ([System.Drawing.FontStyle]::Bold)

    Draw-FittedTextBlock $graphics ([string]$card.eyebrow) 64 35 800 45 24 18 ([System.Drawing.FontStyle]::Bold) $accentBrush
    $titleWidth = if ($card.badge) { 810 } else { 1156 }
    Draw-FittedTextBlock $graphics ([string]$card.title) 62 90 $titleWidth 145 52 38 ([System.Drawing.FontStyle]::Bold) $ink
    if ($card.body) {
        $bodyHeight = if ($card.evidence) { 285 } else { 335 }
        Draw-FittedTextBlock $graphics (($card.body -join "`n")) 76 265 1120 $bodyHeight 31 24 ([System.Drawing.FontStyle]::Regular) $ink
    }
    if ($card.evidence) {
        Draw-FittedTextBlock $graphics ([string]$card.evidence) 76 572 1120 40 18 15 ([System.Drawing.FontStyle]::Regular) $muted
    }
    if ($card.badge) {
        $badgeBrush = switch ($card.badge_kind) { "danger" { $red } "warning" { $orange } "success" { $green } default { $blue } }
        $badgeRect = [System.Drawing.RectangleF]::new(932, 32, 286, 54)
        $graphics.FillRectangle($badgeBrush, $badgeRect)
        $badgeFormat = [System.Drawing.StringFormat]::new()
        $badgeFormat.Alignment = [System.Drawing.StringAlignment]::Center
        $badgeFormat.LineAlignment = [System.Drawing.StringAlignment]::Center
        $graphics.DrawString([string]$card.badge, $badgeFont, $white, $badgeRect, $badgeFormat)
        $badgeFormat.Dispose()
    }
    Draw-FittedTextBlock $graphics ([string]$card.footer) 64 657 1150 38 21 16 ([System.Drawing.FontStyle]::Regular) $muted

    $bitmap.Save($target, [System.Drawing.Imaging.ImageFormat]::Png)
    $badgeFont.Dispose()
    $ink.Dispose(); $muted.Dispose(); $blue.Dispose(); $orange.Dispose(); $red.Dispose(); $green.Dispose(); $white.Dispose(); $panel.Dispose()
    $graphics.Dispose(); $bitmap.Dispose()
}

$privateFonts.Dispose()
