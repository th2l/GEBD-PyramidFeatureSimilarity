FROM tensorflow/tensorflow:2.9.1-gpu
RUN pip install tensorflow==2.9.1 tf-models-official==2.9.2 tensorflow-io==0.26.0 scipy==1.8.1 tqdm==4.64.0 yacs==0.1.8 contextlib2==21.6.0 wandb --no-cache-dir
