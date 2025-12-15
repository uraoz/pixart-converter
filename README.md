ドット絵ぽくするやつ

"-o", "--output", default="output.png"
Output filename
"-p", "--pixel-size", type=int, default=1
Pixel size for downscaling
"-b", "--brightness", type=float, default=0
Brightness: -100 to 200
"-c", "--contrast", type=float, default=0,
Contrast: -100 to 200
"-s", "--saturation", type=float, default=0,
Saturation: -100 to 200
"--palette", default="default",
Palette name
"--no-palette", action="store_true",
Disable palette mapping
"--scale-up", action="store_true",
Scale output back to original size
"--ciede2000", action="store_true",
Use CIEDE2000 color distance
"--list-palettes", action="store_true",
List available palettes and exit
    
