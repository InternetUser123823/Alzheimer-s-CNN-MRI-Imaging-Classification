# CNN Development Log

## Version 1

### Architecture
- 3 convolutional blocks (32-64-128 filters)
- Adam (lr=0.001)
- Weighted CrossEntropy
- Batch size = 32

### Result
**Best Model**

`Epoch 14/20 | Train loss: 1.4061, acc: 0.3444 | Val loss: 1.3544, acc: 0.5010`

### Conclusions
- Suspected that the model predicts the majority class most of the time.
- This could be due to too-aggressive class weights combined with a too-high learning rate, which could contribute to large, destabilizing gradient updates when a ModerateDemented image is included in a batch and processed.

---

## Version 2

### Changes
- Adam learning rate reduced by 10x (lr=0.0001)
- Implemented gradient clipping to rescale all gradients in case gradients are too large after backpropagation.
- Implemented AdaptiveAvgPool2d to reduce the number of parameters in the fully connected layers, reducing model complexity.
- Implemented BatchNorm2d to stabilize activations between layers.

### Result
**Best Model**

`Epoch 18/20 | Train loss: 1.1614, acc: 0.5416 | Val loss: 1.1105, acc: 0.5752`

### Conclusions
- Validation accuracy slightly improved, yet is still oscillating at approximately 57% at best.
- Training accuracy is steadily improving.
- Validation loss is approximately 1.08.
- Suspected that the rare frequency of the ModerateDemented cases could be the root of the issue, combined with the high class weight for ModerateDemented.

---

## Version 3

### Changes

- Changed random split to stratified split between validation and training so that the model can see about the same percentage of ModerateDemented cases in both groups; thus, the oscillation in validation accuracy can be reduced.
- Implemented square-root weighted random sampling to ensure that each class has more of an equal chance of being selected for the batch.
- Implemented confusion matrix and F1-score conclusions to further pinpoint issues with the model.
- Doubled the number of filters in each block; increased the first block to 96 filters (`96 → 128 → 256`).

### Result

**Best model — Epoch 18/20**

| Metric | Training | Validation |
|---|---:|---:|
| Loss | 0.9360 | 0.8561 |
| Accuracy | 0.5571 | 0.6039 |

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| MildDemented | 0.781 | 0.174 | 0.284 | 144 |
| ModerateDemented | 0.667 | 1.000 | 0.800 | 10 |
| NonDemented | 0.724 | 0.719 | 0.722 | 512 |
| VeryMildDemented | 0.460 | 0.602 | 0.521 | 359 |
| **Accuracy** |  |  | **0.604** | 1,025 |
| **Macro average** | 0.658 | 0.624 | **0.582** | 1,025 |
| **Weighted average** | 0.639 | 0.604 | **0.591** | 1,025 |

**Confusion matrix**  
*Rows = true classes; columns = predicted classes.*

```text
[[ 25,   1,   6, 112],
 [  0,  10,   0,   0],
 [  1,   1, 368, 142],
 [  6,   3, 134, 216]]
```

### Conclusions

- Validation accuracy slightly improved, yet was still oscillating at approximately 60% at best.
- Training accuracy steadily improved, yet the highest accuracy increased only a little compared with the highest accuracy of the previous model version, approximately 0.02.
- Validation loss was approximately 0.84, a decrease of about 0.17.
- According to the confusion matrix, VeryMildDemented precision was compromised: the model misidentified 142 NonDemented cases as VeryMildDemented. There was also a relatively large frequency of MildDemented cases misidentified as VeryMildDemented.
- The model's macro F1 score improved to 0.582, indicating a substantial improvement in balanced four-class performance despite only a modest improvement in overall accuracy.

---

## Version 4

### Changes

- Added a scheduler to decrease the learning rate if validation loss did not decrease within three epochs.
- Tested adding “soft” weights: class weights that were the normalized inverse square root of the number of cases in each training class. These were removed because the soft weights compounded with the scheduler became too aggressive.
- Added weight decay to the optimizer to regularize training and reduce overfitting by penalizing large weight values on every update.
- Adjusted the code so that the best model was selected based on the highest validation F1 score attained.
- Fixed F1 training-score reporting. The training F1 score was now based on the original training dataset rather than the augmented training dataset, which required a new loader.
- Tested 50 epochs.

### Result

**Best model — Epoch 40/50**

| Metric | Training | Validation |
|---|---:|---:|
| Loss | 0.8243 | 0.7697 |
| Accuracy | 0.6226 | 0.6341 |
| Macro F1 | 0.6838 | 0.6617 |

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| MildDemented | 0.537 | 0.403 | 0.460 | 144 |
| ModerateDemented | 0.833 | 1.000 | 0.909 | 10 |
| NonDemented | 0.752 | 0.748 | 0.750 | 512 |
| VeryMildDemented | 0.503 | 0.554 | 0.527 | 359 |
| **Accuracy** |  |  | **0.634** | 1,025 |
| **Macro average** | 0.656 | 0.676 | **0.662** | 1,025 |
| **Weighted average** | 0.635 | 0.634 | **0.633** | 1,025 |

**Confusion matrix**  
*Rows = true classes; columns = predicted classes.*

```text
[[ 58,   0,   4,  82],
 [  0,  10,   0,   0],
 [ 13,   1, 383, 115],
 [ 37,   1, 122, 199]]
```

### Conclusions

- When 50 epochs were tested, it was found that although the first approximately 30 epochs demonstrated an oscillating validation F1 score, validation F1 and validation accuracy plateaued near the final approximately 10 epochs.
- This suggested that the oscillation problem may have been due to an insufficient number of opportunities for the model to see the images.
- Further work would focus on distinguishing VeryMildDemented from NonDemented.

---

## Version 5

### Changes

- Added early stopping so that if validation F1 did not increase within seven consecutive epochs, model training would stop.
- Implemented k-fold validation. Thus, validation F1 scores would be more reliable across different validation splits rather than based on only one split.
- Tested the following modifications separately. Modifications with an asterisk (`*`) substantially improved validation F1 scores and were implemented in the final version:
  - Label smoothing: can help the model generalize better, particularly when classes are difficult to distinguish, such as VeryMildDemented and NonDemented, by encouraging the model to be less confident about certain classes.
  - Removed jitter from augmentation: variation in brightness and contrast caused by jitter could have concealed or altered subtle differences between VeryMildDemented and NonDemented.
  - Focal loss*: the model's differentiation between VeryMildDemented and NonDemented could benefit from increasing loss associated with more challenging images and reducing loss associated with easier images.
  - Added one more block*: added a 256-filter block with no pooling after the third block. This gave the model greater capacity to learn higher-level and more abstract features, which could help distinguish subtle visual differences between VeryMildDemented and NonDemented images.
  - Implemented adaptive max pooling: allowed feature maps to be downsampled more gradually, retaining more spatial information and potentially preserving subtle features while avoiding the computational and memory demands of maintaining larger feature maps.
- Added the previously unseen testing set for evaluation such that the model performance could be reliably gauged at the end; additionally, implemented a variable epoch count based on the average of the sum of epochs it takes for all folds.
  
### Cross-Validation Result

| Fold | Best Epoch | Train Loss | Train Accuracy | Train F1 | Validation Loss | Validation Accuracy | Validation F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34/50 | 0.0424 | 0.9595 | 0.9942 | 0.0680 | 0.9405 | 0.9609 |
| 2 | 19/50 | 0.0785 | 0.9329 | 0.9760 | 0.1092 | 0.9062 | 0.9257 |
| 3 | 40/50 | 0.0413 | 0.9639 | 0.9935 | 0.0656 | 0.9414 | 0.9546 |
| 4 | 36/50 | 0.0458 | 0.9590 | 0.9933 | 0.0805 | 0.9355 | 0.9489 |
| 5 | 33/50 | 0.0644 | 0.9443 | 0.9893 | 0.0761 | 0.9346 | 0.9523 |

**Macro F1:**

\[
0.9485 \pm 0.0135
\]

### Best Validation Fold

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| MildDemented | 0.979 | 0.986 | 0.983 | 144 |
| ModerateDemented | 1.000 | 1.000 | 1.000 | 10 |
| NonDemented | 0.940 | 0.947 | 0.944 | 512 |
| VeryMildDemented | 0.924 | 0.911 | 0.917 | 359 |
| **Accuracy** |  |  | **0.940** | 1,025 |
| **Macro average** | 0.961 | 0.961 | **0.961** | 1,025 |
| **Weighted average** | 0.940 | 0.940 | **0.940** | 1,025 |

**Confusion matrix**  
*Rows = true classes; columns = predicted classes.*

```text
[,[0][2][142]
 [  0,  10,   0,   0],
 [  2,   0, 485,  25],
 [  1,   0,  31, 327]]
```

### Final Test Result

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| MildDemented | 0.855 | 0.296 | 0.440 | 179 |
| ModerateDemented | 1.000 | 0.333 | 0.500 | 12 |
| NonDemented | 0.865 | 0.562 | 0.682 | 640 |
| VeryMildDemented | 0.508 | 0.904 | 0.651 | 448 |
| **Accuracy** |  |  | **0.643** | 1,279 |
| **Macro average** | 0.807 | 0.524 | **0.568** | 1,279 |
| **Weighted average** | 0.740 | 0.643 | **0.635** | 1,279 |

**Confusion matrix**  
*Rows = true classes; columns = predicted classes.*

```text
[[ 53,   0,  15, 111],
 [  2,   4,   2,   4],
 [  3,   0, 360, 277],
 [  4,   0,  39, 405]]
```

### Conclusions

- The large abnormal gap between the validation F1 score and the test F1 score indicated that the model generalized poorly from the training and validation sets to the test set.
- Because the training and validation F1 scores did not differ substantially, the results may suggest differences between the training/validation and test data distributions or other dataset characteristics that affected generalization, such as multiple scans belonging to the same patient.
- Further work should focus on using OASIS data so that patient characteristics can also be considered.
- Transfer learning could also be implemented to enhance the model's capability in classifying different stages of Alzheimer's disease.
