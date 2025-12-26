# https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/
# modified by ablanco50, Alfonso Blanco
# dataset  https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24

from rfdetr import RFDETRBase


import torch
# https://stackoverflow.com/questions/75517324/runtimeerror-inference-tensors-cannot-be-saved-for-backward-to-work-around-you
torch.inference_mode=torch.no_grad


def main():
        model = RFDETRBase()
        history = []

        def callback2(data):
                history.append(data)

        model.callbacks["on_fit_epoch_end"].append(callback2)
        
        model.train(dataset_dir="Aerial Maritime.v24i.coco", epochs=20, batch_size=16, lr=1e-4)
        
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(history)

        plt.figure(figsize=(12, 8))

        plt.plot(
                df['epoch'],
                df['train_loss'],
                label='Training Loss',
                marker='o',
                linestyle='-'
        )

        plt.plot(
                df['epoch'],
                df['test_loss'],
                label='Validation Loss',
                marker='o',
                linestyle='--'
        )

        plt.title('Train/Validation Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.show()
       
if __name__ == "__main__":
    main()            
