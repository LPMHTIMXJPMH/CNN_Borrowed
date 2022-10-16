import methods

def non_max_suppression(
    bboxes,
    thre,
):
    assert type(bboxes) == list
    # bboxes = [[index, score, x1, y1, x2, y2]]
    bboxes = [box for box in bboxes if box[1] > thre]
    nms_bboxes = []
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)

    while bboxes:
        first_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != first_box[0] or \
            methods.iou(first_box[1:], box[1:])[1] < thre
        ]
        nms_bboxes.append(first_box)
    
    return nms_bboxes

def nms_nms_nms():
    nms_bboxes = non_max_suppression([[0,0.8,300, 400, 800, 900],[0,0.9,400, 400, 800, 800],[1,0.7,200,300,400,500]], 0.6, )
    print(nms_bboxes)
# nms_nms_nms()
