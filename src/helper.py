import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import sys


IM_SHAPE = (64, 64, 3)


def plot_image_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(np.squeeze(img), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )



def plot_value_prediction(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([_ for _ in range(10)])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")
    
    return (i, predicted_label) if predicted_label != true_label else None


class TrainingDatasetLoader(object):
    def __init__(self, data_path, channels_last=True):
        print("Opening {}".format(data_path))
        sys.stdout.flush()

        self.cache = h5py.File(data_path, "r")

        print("Loading data into memory...")
        sys.stdout.flush()
        self.images = self.cache["images"][:]
        self.channels_last = channels_last
        self.labels = self.cache["labels"][:].astype(np.float32)
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]

        self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]

    def get_train_size(self):
        return self.train_inds.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size() // factor // batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            selected_inds = np.random.choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos
            )
        else:
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=n // 2, replace=False, p=p_pos
            )
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=n // 2, replace=False, p=p_neg
            )
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :, :, ::-1] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]

        if not self.channels_last:
            train_img = np.ascontiguousarray(
                np.transpose(train_img, (0, 3, 1, 2))
            )  # [B, H, W, C] -> [B, C, H, W]
        return (
            (train_img, train_label, sorted_inds)
            if return_inds
            else (train_img, train_label)
        )

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[: 10 * n : 10]]
        return (self.images[most_prob_inds, ...] / 255.0).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]




class MyImageDataset(Dataset):
    def __init__(self, image_folder, labels=None, transform=None):
        self.image_folder = Path(image_folder)
        self.labels = labels
        self.transform = transform

        self.image_path_list = sorted(self.image_folder.glob("*.png"))
        # 15th character in path string is the label digit
        if not self.labels:
            self.labels = [int(str(img_path)[15]) for img_path in self.image_path_list]

    def get_path(self, index):
        return self.image_path_list[index]
    
    def get_label(self, index):
        return self.labels[index]
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, index):
        img_path = self.image_path_list[index]

        image = Image.open(img_path).convert("L")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        image = self.normalize_digit(image)


        return image, label

    def normalize_digit(self, img, target_size=20):
            # img shape: (1, 28, 28)
            img = img.squeeze(0)
    
            # Find digit pixels
            mask = img > 0.1
            rows, cols = torch.where(mask)
    
            if len(rows) == 0:
                return img.unsqueeze(0)
    
            # Crop bounding box
            cropped = img[
                rows.min():rows.max()+1,
                cols.min():cols.max()+1
            ]
    
            h, w = cropped.shape
    
            # Preserve aspect ratio
            scale = target_size / max(h, w)
    
            new_h = max(1, round(h * scale))
            new_w = max(1, round(w * scale))
    
            resized = torch.nn.functional.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze()
    
            # Place inside 28x28 canvas
            canvas = torch.zeros((28, 28))
    
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
    
            canvas[top:top+new_h, left:left+new_w] = resized
    
            return canvas.unsqueeze(0)

    