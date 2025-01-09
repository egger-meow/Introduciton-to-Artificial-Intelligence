from PIL import Image, ImageDraw

img = Image.open('data/image.png')

draw = ImageDraw.Draw(img)

bounds = open('data/bounding_box.txt')

boxes = bounds.readlines()

for bounds in boxes:
    bs = list(map(lambda x:int(x), bounds.split(' ')))
    draw.rectangle([(bs[0], bs[1]), (bs[2], bs[3])], outline = 'red', width = 3)
    
img.save('test.png')
    

