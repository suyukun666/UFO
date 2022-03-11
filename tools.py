import numpy as np

def custom_print(context, log_file, mode):
    #custom print and log out function
    if mode == 'w':
        fp = open(log_file, mode)
        fp.write(context + '\n')
        fp.close()
    elif mode == 'a+':
        print(context)
        fp = open(log_file, mode)
        print(context, file=fp)
        fp.close()
    else:
        raise Exception('other file operation is unimplemented !')


def generate_binary_map(pred, type,th=0.5):
    if type == '2mean':
        threshold = np.mean(pred) * 2
        if threshold > th:
            threshold = th
        binary_map = pred > threshold
        return binary_map.astype(np.float32)

    if type == 'mean+std':
        threshold = np.mean(pred) + np.std(pred)
        if threshold > th:
            threshold = th
        binary_map = pred > threshold
        return binary_map.astype(np.float32)



def calc_precision_and_jaccard(pred, gt,th=0.5):
    bin_pred = generate_binary_map(pred, 'mean+std',th)
    tp = (bin_pred == gt).sum()
    precision = tp / (pred.size)

    i = (bin_pred * gt).sum()
    u = bin_pred.sum() + gt.sum() - i
    jaccard = i / (u + 1e-10)

    return precision, jaccard