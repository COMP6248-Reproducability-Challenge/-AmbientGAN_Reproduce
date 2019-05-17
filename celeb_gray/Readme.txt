This folder includes codes for section 2.2.1 in the reproducing paper

Follow the steps below to run code in this folder.

For generating results of Ambient gan

1. Put the dataset img_align_celeba in the folder ./data

2. Run “train.py”

For generating results of baseline (Pure DCGAN)

1. Put the dataset img_align_celeba in the folder ./data

2. Replace the sentence in “ambientGAN.py”
   “self.Y_g = self.measurement_fn(self.X_g, name="measurement_fn”)”
   With
   “self.Y_g = self.X_g”

3. Run “train.py”