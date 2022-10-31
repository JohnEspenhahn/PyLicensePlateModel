
def draw_detections(diskpath, detections):
    output_img = cv2.imread(diskpath)
    x, y, xmax, ymax = detections['boxes'][0]
    w = xmax - x
    h = ymax - y
    color = (39, 129, 113)
    cv2.rectangle(output_img, (x, y), (xmin + w, ymin + h), color, 2)
    cv2.imwrite("output.jpg", output_img)
    