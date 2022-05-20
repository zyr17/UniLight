FROM zyr17/cityflow
RUN pip install --no-cache-dir dataclasses==0.8 numpy==1.19.4 Pillow==8.0.1 protobuf==3.14.0 PyYAML==5.3.1 pyyaml-include==1.2.post2 six==1.15.0 tensorboardX==2.1 torch==1.7.1 torchvision==0.8.2 typing-extensions==3.7.4.3 wandb==0.12.0 -i https://pypi.douban.com/simple/
