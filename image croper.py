from PIL import Image

img = Image.open("original_digit.png").convert("L")
img = img.resize((28, 28))
img.save("resized_digit.png")
