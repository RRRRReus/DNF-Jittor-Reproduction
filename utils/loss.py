from jittor import nn 
from jittor.nn import MSELoss, L1Loss 

class Losses(nn.Module):
    def __init__(self, classes, names, weights, positions, gt_positions):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.names = names
        self.weights = weights
        self.positions = positions
        self.gt_positions = gt_positions
        for class_name in classes:
            # eval()会查找当前作用域中的MSELoss和L1Loss，由于我们从jittor导入了它们，所以这里能正常工作
            module_class = eval(class_name)
            self.module_list.append(module_class())

    def __len__(self):
        return len(self.names)

    def execute(self, outputs, targets):
        losses = []
        for i in range(len(self.names)):
            loss = self.module_list[i](outputs[self.positions[i]], targets[self.gt_positions[i]]) * self.weights[i]
            losses.append(loss)
        return losses

def build_loss(config):
    # 这个工厂函数与框架无关，无需改动
    loss_names = config['types']
    loss_classes = config['classes']
    loss_weights = config['weights']
    loss_positions = config['which_stage']
    print("!!! DEBUG: The loss is actually built with which_stage:", loss_positions)
    loss_gt_positions = config['which_gt']
    assert len(loss_names) == len(loss_weights) == \
           len(loss_classes) == len(loss_positions) == \
           len(loss_gt_positions)
    criterion = Losses(classes=loss_classes, names=loss_names,
                       weights=loss_weights, positions=loss_positions,
                       gt_positions=loss_gt_positions)
    return criterion