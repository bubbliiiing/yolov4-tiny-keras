#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body

if __name__ == "__main__":
    input_shape     = [416, 416, 3]
    anchors_mask    = [[3, 4, 5], [1, 2, 3]]
    num_classes     = 80

    model = yolo_body(input_shape, anchors_mask, num_classes, phi = 0)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
