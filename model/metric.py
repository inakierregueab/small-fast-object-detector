import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def class_accuracy(output, target, threshold=0.5):
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    num_heads = 3
    with torch.no_grad():
        for i in range(num_heads):
            # target[i] = target[i].to(config.DEVICE)
            obj = target[i][..., 0] == 1
            noobj = target[i][..., 0] == 0

            correct_class += torch.sum(torch.argmax(output[i][..., 5:][obj], dim=-1) == target[i][..., 5][obj])
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(output[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == target[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        class_acc = (correct_class / (tot_class_preds + 1e-16)) * 100
        noobj_acc = (correct_noobj / (tot_noobj + 1e-16)) * 100
        obj_acc = (correct_obj / (tot_obj + 1e-16)) * 100

        print(f"Class accuracy is: {class_acc:2f}%")
        print(f"No obj accuracy is: {noobj_acc:2f}%")
        print(f"Obj accuracy is: {obj_acc:2f}%")

    return (class_acc, noobj_acc, obj_acc)
