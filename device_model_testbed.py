import warnings

warnings.filterwarnings("ignore")
from code_deepQueueNet import deviceModel
from code_deepQueueNet.config import RouterConfig
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessing", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

plt.style.use("ggplot")


print("Running device model...")
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
if not args.gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = RouterConfig()
if args.train:
    model = deviceModel.deepQueueNet(
        config, target=["time_in_sys"], data_preprocessing=args.preprocessing
    )  # please turn it on when you run the cell for the first time
    model.build_and_training()

# Learning curve
if args.eval:
    from code_deepQueueNet.config import modelConfigTestbed
    from code_deepQueueNet import eval_metrics

    ins = eval_metrics.REPO(RouterConfig(), modelConfigTestbed(), target=["time_in_sys"])
    ins.loadModel_and_Eval()
    ins.learning_curve()
    ins.regression_rho()
    ins.distrib()

    # import scipy.stats as measures
    from scipy.stats import wasserstein_distance

    y = ins.y1
    y_pred = ins.y1_pred
    b1 = [0] * len(y)
    print(
        "\tW1/ground truth (deepqueuenet): {}".format(
            wasserstein_distance(y, y_pred) / wasserstein_distance(b1, y)
        )
    )
