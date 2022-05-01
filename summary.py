#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [416, 416, 3]
    anchors_mask    = [[3, 4, 5], [1, 2, 3]]
    num_classes     = 80
    phi             = 0

    model = yolo_body([input_shape[0], input_shape[1], 3], anchors_mask, num_classes, phi=phi)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
