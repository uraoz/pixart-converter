ドット絵ぽくするやつ
```
"-o", "--output", default="output.png"
#Output filename

"-p", "--pixel-size", type=int, default=1
#Pixel size for downscaling

"-b", "--brightness", type=float, default=0
#Brightness: -100 to 200

"-c", "--contrast", type=float, default=0,
#Contrast: -100 to 200

"-s", "--saturation", type=float, default=0,
#Saturation: -100 to 200

"--palette", default="default",
#Palette name

"--no-palette", action="store_true",
#Disable palette mapping

"--scale-up", action="store_true",
#Scale output back to original size

"--ciede2000", action="store_true",
#Use CIEDE2000 color distance

"--list-palettes", action="store_true",
#List available palettes and exit
```
example input
<img width="1280" height="720" alt="bg_title" src="https://github.com/user-attachments/assets/7d814abf-1892-45b3-a67e-ccec1f974dc0" />
example output
<img width="1280" height="720" alt="output" src="https://github.com/user-attachments/assets/ac6cc2ea-994d-43e8-b6ba-52194d644f7b" />
