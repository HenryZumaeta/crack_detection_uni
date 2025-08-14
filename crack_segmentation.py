# Librerías
import albumentations as A
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import torch
import numpy as np
from PIL import Image
import os

# Dataset básico con aumentación
class CrackDataset(Dataset):
    def __init__(self, img_paths, msk_paths, feature_extractor, augment=False):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.aug = A.Compose([A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)])

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        msk = (np.array(Image.open(self.msk_paths[idx]))[..., :3] == [255,0,255]).all(-1).astype(np.uint8)
        if self.augment: msk = self.aug(image=img, mask=msk)['mask']
        inputs = self.feature_extractor(img, segmentation_maps=msk, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def __len__(self):
        return len(self.img_paths)

# Carga de datos
img_dir = "data/CrackIPN/images"
msk_dir = "data/CrackIPN/masks"
img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
msk_files = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir)])

# División 70/15/15
from sklearn.model_selection import train_test_split
train_img, test_val_img, train_msk, test_val_msk = train_test_split(img_files, msk_files, test_size=0.3, random_state=42)
val_img, test_img, val_msk, test_msk = train_test_split(test_val_img, test_val_msk, test_size=0.5, random_state=42)

# Modelos
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b5", size={"height": 320, "width": 480})
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", num_labels=2).to("cuda")

# Métrica principal
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    upsampled = torch.nn.functional.interpolate(torch.from_numpy(logits), size=labels.shape[-2:], mode='bilinear')
    preds = torch.argmax(upsampled, dim=1).flatten().numpy()
    return {"f1": f1_score(labels.flatten(), preds, average="binary")}

# Entrenamiento
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=16,
        num_train_epochs=150,
        fp16=True,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    ),
    train_dataset=CrackDataset(train_img, train_msk, feature_extractor, augment=True),
    eval_dataset=CrackDataset(val_img, val_msk, feature_extractor),
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate(CrackDataset(test_img, test_msk, feature_extractor))
print(f"F1-score en prueba: {results['eval_f1']:.4f}")
