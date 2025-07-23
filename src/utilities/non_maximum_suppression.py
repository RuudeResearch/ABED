
"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""

import torch


def non_maximum_suppression(localizations, scores, overlap=0.5, top_k=200):
    """1D nms"""
    x = localizations[:, 0]
    y = localizations[:, 1]

    areas = y - x
    order = scores.sort(0, descending=True)[1][:top_k]

    keep = []
    while order.numel() > 1:
        i = order[0]
        keep.append([x[i], y[i], scores[i], i])#, allscores[i,0], allscores[i,1], allscores[i,2]])
        order = order[1:]
        xx = torch.clamp(x[order], min=x[i])
        yy = torch.clamp(y[order], max=y[i])

        intersection = torch.clamp(yy - xx, min=0)

        intersection_over_union = intersection / \
            (areas[i] + areas[order] - intersection)

        order = order[intersection_over_union <= overlap]

    # remaining element if order has size 1
    keep.extend([[x[k], y[k], scores[k], k]for k in order])#, allscores[k,1], allscores[k,2]] for k in order])


    return keep


if __name__ == "__main__":
    localizations_scores = torch.FloatTensor(
        [
            [20, 50, 0.99],  # 0
            [10, 50, 0.97],  # 1
            [25, 42, 0.6],   # 2
            [30, 60, 0.98],  # 3

            [75, 85, 0.92],  # 4
            [72, 87, 0.90],  # 5
            [76, 85, 0.78],  # 6
            [80, 90, 0.91],  # 7
        ]
    )
    all_scores = torch.FloatTensor(
        [
            [0.11, 0.11, 0.99],  # 0
            [0.21, 0.12, 0.97],  # 1
            [0.12, 0.21, 0.6],   # 2
            [0.31, 0.13, 0.98],  # 3

            [0.32, 0.23, 0.92],  # 4
            [0.41, 0.14, 0.90],  # 5
            [0.14, 0.41, 0.78],  # 6
            [0.42, 0.24, 0.91],  # 7
        ]
    )
    localizations = localizations_scores[:, :2] / 100
    scores = localizations_scores[:, -1]
    overlap = 0.5
    top_k = 30
    keep = non_maximum_suppression(localizations, scores, overlap, top_k)
    
    '''
    localizations_decoded_selected = torch.floatTensor(
                                    [[0.0558, 0.1410],
                                     [0.0543, 0.1395],
                                     [0.0539, 0.1404]])
    scores_batch_class_selected = torch.floatTensor(
                                    [0.8212, 
                                     0.8356, 
                                     0.8261])
    
    non_maximum_suppression(
                            localizations_decoded_selected,
                            scores_batch_class_selected,
                            scores[i, :, :].data,
                            overlap=self.overlap_non_maximum_suppression,
                            top_k=self.top_k_non_maximum_suppression)
    '''
