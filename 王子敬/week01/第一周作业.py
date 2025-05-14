# 环境检查代码
# 环境运行在conda中py312之下

python --version
python -c "import torch; print(f'\nPyTorch版本: {torch.__version__}\nCUDA是否可用: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'\nTransformers版本: {transformers.__version__}')"
python -c "import peft; print(f'\nPEFT版本: {peft.__version__}')"
python -c "import sklearn; print(f'\nScikit-learn版本: {sklearn.__version__}')"
python -c "import numpy; print(f'\nNumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'\nPandas版本: {pandas.__version__}')"
