yolo = [
    (7,64,2,3),
    "maxpool",
    (3,192,1,1),
    "maxpool",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "maxpool",
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,512,1,0),
    (3,1024,1,1),
    "maxpool",
    (1,512,1,0),
    (3,1024,1,1),
    (1,512,1,0),
    (3,1024,1,1),
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]

loss = dict()

loss['qutity'] = 7*7
loss['split'] = 7
loss['num_cls'] = 20
# guesser guesses bounding box position(4)
# and if there is an object's center point
# in the guesser box(1) from all
# loss['qutity'] boxes."loss['guesser']" times
loss['guesser'] = (4 + 1) * 2
