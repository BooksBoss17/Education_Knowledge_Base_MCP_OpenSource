param([Parameter(Mandatory)][string]$Source,[Parameter(Mandatory)][string]$Output)
$ErrorActionPreference='Stop'
$existing=@(Get-Process wps,wpp -ErrorAction SilentlyContinue | ForEach-Object Id)
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class StudioWindow { [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd,out uint id); }'
Add-Type -TypeDefinition ([IO.File]::ReadAllText((Join-Path (Split-Path $env:ST_SCRIPT) 'StudioMatte.cs'))) -ReferencedAssemblies System.Drawing
$app=$null
$deck=$null
try {
    $app=New-Object -ComObject KWPP.Application
    [uint32]$owner=0
    try {[void][StudioWindow]::GetWindowThreadProcessId([IntPtr]$app.HWND,[ref]$owner)} catch {}
    try {$app.AutomationSecurity=3} catch {}
    $deck=$app.Presentations.Open($Source,-1,0,0)
    $w=[double]$deck.PageSetup.SlideWidth
    $h=[double]$deck.PageSetup.SlideHeight
    $ph=[int][Math]::Round(1920*$h/$w)
    $pages=@()
    foreach($slide in $deck.Slides){
        $n=$slide.SlideIndex
        $ref="page-$n.png"
        $slide.Export((Join-Path $Output $ref),'PNG',1920,$ph)
        $items=@()
        $visible=@()
        for ($idx=1;$idx -le $slide.Shapes.Count;$idx++) {
            $shape=$slide.Shapes.Item($idx)
            $originalVisibility=[int]$shape.Visible
            $sid=[string]$shape.Id
            $layer="object-$n-$sid.png"
            $o=@{id="s$n-$sid";name=[string]$shape.Name; x=[double]$shape.Left;y=[double]$shape.Top;w=[Math]::Max(1,[double]$shape.Width);h=[Math]::Max(1,[double]$shape.Height);rotation=[double]$shape.Rotation;z=[int]$shape.ZOrderPosition;type=[int]$shape.Type;text='';fontSize=18}
            try {if($shape.HasTextFrame -and $shape.TextFrame.HasText){$o.text=[string]$shape.TextFrame.TextRange.Text;$v=$shape.TextFrame.TextRange.Font.Size;if($v -gt 0){$o.fontSize=[double]$v}}} catch {}
            try {$o.geometry=[int]$shape.AutoShapeType} catch {}
            try {
                if ($shape.HasTable) {
                    $rows=@()
                    for($row=1;$row -le $shape.Table.Rows.Count;$row++){
                        $cells=@()
                        for($col=1;$col -le $shape.Table.Columns.Count;$col++){
                            $cells += [string]$shape.Table.Cell($row,$col).Shape.TextFrame.TextRange.Text
                        }
                        $rows+=($cells -join [char]9)
                    }
                    $o.text=$rows -join [Environment]::NewLine
                }
            } catch {}
            try {
                if($shape.HasTextFrame -and $shape.TextFrame.HasText){
                    $font=$shape.TextFrame.TextRange.Font
                    $o.font=[string]$font.Name
                    $o.bold=([int]$font.Bold -eq -1)
                    $rgb=[int]$font.Color.RGB
                    if($rgb -ge 0){$o.color=('#{0:X2}{1:X2}{2:X2}' -f ($rgb -band 255),(($rgb -shr 8) -band 255),(($rgb -shr 16) -band 255))}
                }
            } catch {}
            $items+= $o
            $visible+=@{shape=$shape;original=$originalVisibility;item=$o;layer=$layer}
        }
        $bg="background-$n.png"
        try {
            foreach($entry in $visible){$entry.shape.Visible=0}
            $slide.Export((Join-Path $Output $bg),'PNG',1920,$ph)
            $slide.FollowMasterBackground=0
            $slide.DisplayMasterShapes=0
            $fill=$slide.Background.Fill
            $fill.Visible=-1
            $fill.Solid()
            $dark=Join-Path $Output 'matte-black.png'
            $light=Join-Path $Output 'matte-white.png'
            foreach($entry in $visible){
                $entry.shape.Visible=-1
                $fill.ForeColor.RGB=0
                $slide.Export($dark,'PNG',1920,$ph)
                $fill.ForeColor.RGB=16777215
                $slide.Export($light,'PNG',1920,$ph)
                $bounds=[StudioMatte]::Save($dark,$light,(Join-Path $Output $entry.layer))
                if($bounds.Count -eq 4){
                    $entry.item.asset=$entry.layer
                    $entry.item.ink=@(($bounds[0]*$w/1920),($bounds[1]*$h/$ph),($bounds[2]*$w/1920),($bounds[3]*$h/$ph))
                }
                $entry.shape.Visible=0
            }
        }finally{foreach($entry in $visible){$entry.shape.Visible=$entry.original}}
        $seq=@()
        try {
            foreach($effect in $slide.TimeLine.MainSequence){
                $step=@{target="s$n-$($effect.Shape.Id)";effect=[int]$effect.EffectType;exit=[bool]$effect.Exit;trigger=[int]$effect.Timing.TriggerType;duration=[double]$effect.Timing.Duration;delay=[double]$effect.Timing.TriggerDelayTime}
                try {$step.direction=[int]$effect.EffectParameters.Direction} catch {}
                try {$step.textUnit=[int]$effect.EffectInformation.TextUnitEffect} catch {}
                if($step.effect -eq 0 -or ($step.effect -ge 86 -and $step.effect -le 149)){
                    foreach($behavior in $effect.Behaviors){try {if($behavior.MotionEffect.Path){$step.path=[string]$behavior.MotionEffect.Path}} catch {}}
                }
                $seq+=$step
            }
        } catch {$seq+=@{error='sequence unavailable'}}
        $transition=0
        try {$transition=[int]$slide.SlideShowTransition.EntryEffect} catch {}
        $seconds=0.3
        try {if($null -ne $slide.SlideShowTransition.Duration){$seconds=[double]$slide.SlideShowTransition.Duration}} catch {}
        $pages+=@{page=$n;reference=$ref;background=$bg;elements=$items;sequence=$seq;transition=$transition;transitionDuration=$seconds}
    }
    @{width=$w;height=$h;slides=$pages}| ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Output 'deck.json') -Encoding UTF8
}finally{
    if($deck){try{$deck.Saved=-1;$deck.Close()}finally{[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck)}}
    if($app){try{if($owner -and $existing -notcontains $owner -and !$app.Presentations.Count){$app.Quit()}}finally{[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)}}
}
