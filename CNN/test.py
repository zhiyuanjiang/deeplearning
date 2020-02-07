from PIL import Image

image = Image.open("F:\\deeplearning-data\\smile-data\\smile.jpg")
image = image.resize((64, 64), Image.ANTIALIAS)
image.show()

