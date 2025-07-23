

"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""
import os


def give_storing_paths(file_name, cd):
    """give_storing_paths

    Parameters:
    -----------
    file_name : string
        current file name
    cd : string
        current directory
    source : string
        source domain
    target : string
        target domain

    Returns:
    --------
    scores_path : string
        where to store the scores
    histories_path : string
        where to store the histories
    weights_path : string
        where to store the weights
    pred_path : string
        where to store the predictions
    """
    print(file_name.split("\\"))
    #print(file_name.split("\\")[-2])
    #print(file_name.split("\\")[-1].split(".")[0])
    phase = file_name.split("\\")[-2]
    expe = file_name.split("\\")[-1].split(".")[0]

    scores_path = os.path.join(cd, "scores", phase)
    weights_path = os.path.join(cd, "weights", phase)
    histories_path = os.path.join(cd, "histories", phase)
    pred_path = os.path.join(cd, "predictions", phase)

    if not os.path.exists(scores_path):
        os.makedirs(scores_path)
    scores_path = os.path.join(scores_path, expe + ".csv")

    if not os.path.exists(histories_path):
        os.makedirs(histories_path)
    histories_path = os.path.join(histories_path, expe + ".csv")

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    weights_path = os.path.join(weights_path, expe)

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    pred_path = os.path.join(pred_path, expe)

    return scores_path, histories_path, weights_path, pred_path
