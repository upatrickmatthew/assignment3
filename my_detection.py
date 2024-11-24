import jetson.inference
import jetson.utils

from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput

net = detectNet("ssd-mobilenet-v2", threshold=0.5)

camera = videoSource("/dev/video0")

display = videoOutput("display://0")


while display.IsStreaming():
	img = camera.Capture()
	
	
	if img is None:
		continue

	detections = net.Detect(img)
	for i in detections:
		print(f"--ClassID:{i.ClassID}")
		print(f"--Confidence:{i.Confidence}")
		print(f"--Left:{i.Left}")
		print(f"--Top:{i.Top}")
		print(f"--Right:{i.Right}")
		print(f"--Bottom:{i.Bottom}")
		print(f"--Width:{i.Width}")
		print(f"--Height:{i.Height}")
		Area = i.Width*i.Height
		Center_x = i.Right-i.Left
		Center_y = i.Bottom - i.Top
		print(f"--Area:{Area}")
		print(f"--Center:{Center_x} {Center_y}")
	display.Render(img)
	display.SetStatus("Objesct Detection | Network {:.0f}FPS".format(net.GetNetworkFPS()))
