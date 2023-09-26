import warnings

warnings.filterwarnings("ignore")
from code_deepQueueNet import deviceModel
from code_deepQueueNet.config import RouterConfig
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessing", action="store_true")
args = parser.parse_args()

plt.style.use("ggplot")


print("Running device model...")
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = RouterConfig()
model = deviceModel.deepQueueNet(
    config, target=["time_in_sys"], data_preprocessing=args.preprocessing
)  # please turn it on when you run the cell for the first time
model.build_and_training()

# Learning curve
from code_deepQueueNet.config import modelConfig
from code_deepQueueNet import eval_metrics

ins = eval_metrics.REPO(RouterConfig(), modelConfig(), target=["time_in_sys"])
ins.loadModel_and_Eval()
ins.learning_curve()
