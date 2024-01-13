from PIL import Image
import os

unpr_data = "Unprocessed_data"
pr_data = "Processed_data" 
target_size = (256, 256)

for filename in os.listdir(unpr_data):
	img_path = os.path.join(unpr_data,filename)
	img = Image.open(img_path)
	img = img.resize(target_size, Image.LANCZOS)
	print("Resizing %s .." %(filename))
	img.save(os.path.join(pr_data,filename))

print("Done")