from PIL import Image

img = Image.open("frontend/originalimg.jpg").convert("L")
img = img.resize((28, 28))
img.save("resized_digit.png")
