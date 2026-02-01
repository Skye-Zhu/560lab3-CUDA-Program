from PIL import Image
img = Image.open("input.png").convert("L")
for M in [512, 1024, 2048]:
    img.resize((M, M)).save(f"input_M{M}.pgm")
print("done")
