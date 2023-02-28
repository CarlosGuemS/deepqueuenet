import warnings  
warnings.filterwarnings("ignore")
from code_deepQueueNet import deviceModel
from code_deepQueueNet.config import BaseConfig 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use('ggplot')


print("Running device model...")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
config=BaseConfig() 
model=deviceModel.deepQueueNet(config, 
                               target=['time_in_sys'], 
                               data_preprocessing=False)  #please turn it on when you run the cell for the first time  
model.build_and_training()

# Learning curve
from code_deepQueueNet.config import modelConfig
from code_deepQueueNet import  eval_metrics
 
ins= eval_metrics.REPO(BaseConfig(), modelConfig(), target=['time_in_sys'])
ins.loadModel_and_Eval()
ins.learning_curve()