import torch
import time
import pandas as pd
import os

from .helper_functions.helper_functions import mAP, AverageMeter
from .models import create_model
from util import cal_score


def ASL(args):
    # setup model
    print('creating model {}...'.format(args.model_type))
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    print('done')
    return model


def asl_validate_multi(val_loader, model, args, res_type=0):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    result = []
    for i, (name, input, target) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(args.threshold).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0
               if tp[i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0
               if tp[i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i])
               if tp[i] > 0 else 0.0
               for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    prec=prec, rec=rec
                )
            )
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

        if res_type == 0:
            temp_cat = []
            cat_id = val_loader.dataset.get_cat2id()
            for item in cat_id:
                temp_cat.append(item)
            for i in range(len(name)):
                labels = []
                labels_gt = []
                for j in range(args.num_classes):
                    if output[i][j] > args.threshold:
                        labels.append(temp_cat[j])
                    if target[i][j] > 0:
                        labels_gt.append(temp_cat[j])
                result.append([name[i].split(os.sep)[-1], "|".join(labels), "|".join(labels_gt)])
        else:
            cat_id = val_loader.dataset.get_cat2id()
            id_cat = list(cat_id.keys())
            for i in range(len(name)):
                temp = {"filename": name[i]}
                labels = []
                for j in range(args.num_classes):
                    if output.numpy()[i][j] > args.threshold:
                        temp[id_cat[j]] = output.numpy()[i][j]
                        labels.append(id_cat[j])
                    else:
                        temp[id_cat[j]] = -1
                temp["labels_gt"] = "|".join(sorted(
                    [id_cat[idx] for idx, value in enumerate(target[i]) if value == 1]
                ))
                temp["labels"] = "|".join(sorted(labels))
                temp["pass"] = 1 if temp["labels_gt"] == temp["labels"] else 0
                result.append(temp)

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    if res_type == 0:
        result = pd.DataFrame(result)
        result.rename(columns={0: "filename", 1: "labels", 2: "labels_gt"}, inplace=True)
        return result, mAP_score
    else:
        result = pd.DataFrame(result)
        result = result[["filename"] + list(sorted(val_loader.dataset.get_cat2id().keys())) + ["labels_gt", "labels", "pass"]]
        accuracy = result.groupby(by="labels_gt", as_index=False, sort=False)[["labels_gt", "pass"]].mean()
        result["score"] = result.apply(
            lambda x: cal_score(
                x["labels_gt"], x["labels"],
                args.num_classes, args.way_num, val_loader.dataset.get_cat2id()
            ), axis=1
        )
        accuracy.rename(columns={"labels_gt": "labels_gt", "pass": "accuracy"}, inplace=True)
        return result, accuracy, mAP_score
