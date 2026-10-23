"""AAT 2: Driver Drowsiness Detection using CNN. All numbers are read from the saved result files."""
import json
from docx_helpers import *

F = "report/figures/"
M = json.load(open("aat2/outputs/metrics.json"))
SYS = json.load(open("aat2/outputs/ddd_system_eval.json"))
FACE = json.load(open("aat2/outputs_face_cnn/metrics.json"))
T, R = M["test"], M["real"]


def pct(v):
    return f"{v * 100:.2f}%"


def fig(doc, path, n, caption, width):
    figure(doc, path, f"Fig {n}: {caption}", width)
    return n + 1


def build(doc, n):
    title(doc, "Driver Drowsiness Detection using CNN")

    heading(doc, "1. Introduction")
    para(doc, f"DriveAlert is a deep learning system that watches a driver through a camera, uses a Convolutional Neural "
              f"Network to decide in every frame whether the driver's eyes are open or closed, and raises an alarm when "
              f"the eyes stay closed for 15 frames in a row. The eye-state CNN reaches {pct(T['accuracy'])} accuracy on "
              f"{T['n']:,} held-out eye images and {pct(R['accuracy'])} on {R['n']:,} real eye photographs from a "
              f"completely different source.")
    para(doc, "Drowsy driving is a major cause of road accidents, especially on highways and at night. A tired driver "
              "reacts slowly, drifts out of the lane and can fall asleep for a few seconds without noticing. Unlike "
              "alcohol, drowsiness cannot be measured with a simple test at the roadside, so it has to be detected while "
              "the person is driving.")
    para(doc, "The most reliable visible sign of drowsiness is long eye closure: an alert person blinks in about a "
              "tenth of a second, while a drowsy person keeps the eyes closed for much longer. DriveAlert therefore "
              "splits the problem into two steps. A CNN classifies each eye as Open or Closed, and a simple frame counter "
              "turns those per-frame decisions into a drowsiness alert. The system runs on an ordinary laptop webcam and "
              "is also available as a Gradio web application.")
    para(doc, "The system is a driver assistance tool and is not a replacement for adequate rest before driving.")

    heading(doc, "2. Objectives")
    para(doc, "The main objective is to build a camera-based system that detects driver drowsiness in real time using a CNN. "
              "The project has the following objectives:")
    bullets(doc, ["**Detect eye state:** classify each eye as Open or Closed with a CNN trained from scratch.",
                  "**Prepare the data:** extract eye regions from face images, normalise them and use augmentation to improve generalisation.",
                  "**Evaluate honestly:** split data by face, and test on real eye photographs from a different source.",
                  "**Detect drowsiness:** raise an alert when the eyes stay closed for 15 consecutive frames.",
                  "**Explain the model:** use Grad-CAM to show which image regions the CNN uses.",
                  "**Provide an interface:** a Gradio web application and a real-time webcam monitor."], numbered=True)
    para(doc, "Expected outcomes", size=12.5, align="left", space_after=3)
    bullets(doc, ["A trained CNN that separates open and closed eyes.",
                  "Training curves, confusion matrices and ROC curves that show how well the model learned.",
                  "A working application that marks a driver as alert or drowsy.",
                  "A clear explanation of why an eye-level model is used instead of a whole-face model."])

    heading(doc, "3. Methodology")
    para(doc, "3.1 Overview of the proposed system", size=12.5, align="left", space_after=3)
    para(doc, "DriveAlert processes every video frame in six stages. The face is found with a Haar cascade detector, two "
              "square eye regions are cut out using the geometry of the face, the CNN gives the probability that the eyes "
              "are closed, and a counter decides when to alert the driver.")
    n = fig(doc, F + "aat2_block_diagram.png", n, "Block diagram of the driver drowsiness detection system", 6.4)

    para(doc, "3.2 Design choice: whole face or eye state", size=12.5, align="left", space_after=3)
    para(doc, f"The first design classified the whole face directly as Drowsy or Non Drowsy using the public Driver "
              f"Drowsiness Dataset (41,793 face images of {FACE['people']} people). To test it fairly, the data was split "
              f"by person, so the test faces were never seen in training. The CNN reached "
              f"{pct(FACE['train_accuracy'])} accuracy on training faces but only {pct(FACE['accuracy'])} on the "
              f"unseen people (AUC {FACE['auc']:.2f}), barely above the {pct(FACE['baseline_always_drowsy'])} obtained by "
              f"always answering Drowsy, and a pre-trained MobileNetV2 feature extractor with a linear classifier reached only "
              f"{pct(FACE['mobilenet_probe_accuracy'])}. With few people, a whole-face model learns who the person is "
              f"instead of how tired they look, and the dataset labels a whole video clip as drowsy even when many frames "
              f"show open eyes.")
    table(doc, [["Approach", "Input", "Accuracy on unseen faces", "Decision"],
                ["CNN on whole face", "96×96 colour face", pct(FACE["accuracy"]), "Rejected"],
                ["MobileNetV2 features on whole face", "128×128 colour face", pct(FACE["mobilenet_probe_accuracy"]), "Rejected"],
                ["CNN on eye state (final)", "64×64 grayscale eye", pct(T["accuracy"]), "Selected"]],
          widths=[2.2, 1.6, 1.5, 1.0])
    para(doc, "The eye-state design was selected because eye closure is a clear, frame-level visual signal that does not "
              "depend on the identity of the driver.")

    para(doc, "3.3 Dataset and eye extraction", size=12.5, align="left", space_after=3)
    para(doc, "Training data comes from the open and closed eyes dataset on Hugging Face, which contains face images with a "
              "bounding box for each eye and a label for the eye state. A sample of 21,099 faces was used. Each eye was cut "
              "out as a square of 1.6 times the eye width, so the crop size does not depend on how open the eye is, and "
              "resized to 64×64 grayscale. Both eyes of one face are always kept in the same split. A separate set "
              "of 3,486 real eye photographs from a different source was kept only for testing.")
    table(doc, [["Attribute", "Value"], ["Source faces", "21,099"], ["Eye crops", f"{sum(M['counts'].values()):,}"],
                ["Train / validation / test", f"{M['counts']['train']:,} / {M['counts']['val']:,} / {M['counts']['test']:,}"],
                ["Real-world test eyes", f"{R['n']:,} ({R['tp'] + R['fn']:,} closed, {R['tn'] + R['fp']:,} open)"],
                ["Input size", "64 × 64 × 1 (grayscale)"], ["Classes", "Open (0), Closed (1)"]], widths=[2.4, 3.4])
    n = fig(doc, F + "aat2_eye_extraction.png", n, "Extraction of the two eye regions from a face image", 5.6)
    n = fig(doc, F + "aat2_dataset_samples.png", n, "Sample eye crops from the training data and the real-world test set", 6.2)

    para(doc, "3.4 Preprocessing and data augmentation", size=12.5, align="left", space_after=3)
    para(doc, "Pixel values are scaled from 0 to 255 to the range 0 to 1. Augmentation is applied only to training images, "
              "so that the CNN sees varied versions of every eye. Blur and noise are applied to both classes, so that "
              "image sharpness cannot be used as a shortcut to the label.")
    table(doc, [["Setting", "Value", "Purpose"], ["Rescale", "1/255", "Normalises pixel values"],
                ["Horizontal flip", "Yes", "Left and right eyes look alike"], ["Rotation", "±8%", "Handles head tilt"],
                ["Zoom / shift", "15% / 8%", "Handles differences in crop position"],
                ["Contrast / brightness", "30% / 20%", "Handles lighting changes in a car"],
                ["Blur (3×3 average)", "35% of images", "Handles motion and out-of-focus frames"],
                ["Gaussian noise (σ = 0.05)", "35% of images", "Handles low-light camera noise"]],
          widths=[1.9, 1.3, 3.0], align=["center", "center", "left"])

    para(doc, "3.5 Model architecture", size=12.5, align="left", space_after=3)
    para(doc, f"The CNN is built from scratch with three convolution blocks. Each block has two 3×3 convolutions "
              f"with batch normalisation and ReLU, followed by max pooling and dropout. Global average pooling and two dense "
              f"layers turn the features into one probability. The model has {M['params']:,} parameters, which is small "
              f"enough to run in real time on a laptop CPU.")
    n = fig(doc, F + "aat2_cnn_architecture.png", n, "Architecture of the eye-state CNN", 6.4)
    table(doc, [["Layer", "Setting", "Output shape"], ["Input", "grayscale eye crop", "64×64×1"],
                ["Conv block 1", "2 × Conv 32, BN, ReLU, MaxPool, Dropout 0.25", "32×32×32"],
                ["Conv block 2", "2 × Conv 64, BN, ReLU, MaxPool, Dropout 0.25", "16×16×64"],
                ["Conv block 3", "2 × Conv 128, BN, ReLU, MaxPool, Dropout 0.25", "8×8×128"],
                ["GlobalAveragePooling2D", "one value per feature map", "128"], ["Dense", "128 units, ReLU", "128"],
                ["Dropout", "0.5", "128"], ["Dense (output)", "1 unit, sigmoid", "P(closed)"]], widths=[1.8, 3.1, 1.3])

    para(doc, "3.6 Training", size=12.5, align="left", space_after=3)
    para(doc, f"The model is trained with the Adam optimiser (learning rate 0.001) and binary cross-entropy loss, with a "
              f"batch size of 128. The learning rate is halved when the validation loss stops improving for 2 epochs, and "
              f"training stops early after 4 epochs without improvement, keeping the best weights. Training ran for "
              f"{M['epochs_run']} epochs on an Apple M1 Pro CPU with TensorFlow 2.20. The batch normalisation momentum was "
              f"set to 0.9 instead of the default 0.99, because with the default value the stored statistics had not "
              f"caught up after the first epochs and the model predicted Closed for every image at inference time.")

    para(doc, "3.7 Drowsiness decision", size=12.5, align="left", space_after=3)
    para(doc, "A single closed-eye frame is normal, because people blink. The system averages P(closed) of the two eyes "
              "and counts consecutive frames with a value of at least 0.5. When 15 frames in a row are closed, which is "
              "about half a second at 30 frames per second and longer than a normal blink, the alarm sounds and a "
              "DROWSINESS ALERT message is shown. Any open-eye frame resets the counter.")

    para(doc, "3.8 System workflow", size=12.5, align="left", space_after=3)
    n = fig(doc, F + "aat2_workflow.png", n, "Training and real-time detection workflow", 5.6)
    bullets(doc, ["**Collect data:** load face images with eye boxes and eye-state labels.",
                  "**Extract eyes:** crop both eyes as 64×64 grayscale images.",
                  "**Split:** 70% training, 15% validation and 15% test, keeping both eyes of a face together.",
                  "**Augment:** apply flip, rotation, zoom, contrast, brightness, blur and noise to training eyes.",
                  "**Train:** fit the CNN with Adam and binary cross-entropy, with early stopping.",
                  "**Evaluate:** measure accuracy, precision, recall, F1 and AUC on the test set and the real-world set.",
                  "**Deploy:** load the model in the Gradio application and the webcam monitor.",
                  "**Detect:** find the face, crop both eyes and predict P(closed) in every frame.",
                  "**Alert:** sound the alarm after 15 consecutive closed-eye frames."], numbered=True)

    heading(doc, "4. Results (With Screenshots)")
    para(doc, "The figures below show the training behaviour, the evaluation of the trained CNN, a system-level test on "
              "real driver videos and the working application.")
    n = fig(doc, F + "aat2_training_curves.png", n, "Training and validation accuracy and loss", 6.4)
    n = fig(doc, F + "aat2_confusion_matrix.png", n, "Confusion matrices on the held-out test set and the real-world set", 6.2)
    para(doc, "Metrics from the confusion matrices", size=12.5, align="left", space_after=3)
    table(doc, [["Measure", "Calculation", "Held-out test", "Real-world"],
                ["True Positive (TP)", "Closed predicted as Closed", f"{T['tp']:,}", f"{R['tp']:,}"],
                ["True Negative (TN)", "Open predicted as Open", f"{T['tn']:,}", f"{R['tn']:,}"],
                ["False Positive (FP)", "Open predicted as Closed", f"{T['fp']:,}", f"{R['fp']:,}"],
                ["False Negative (FN)", "Closed predicted as Open", f"{T['fn']:,}", f"{R['fn']:,}"],
                ["Accuracy", "(TP + TN) / total", f"{T['accuracy']:.4f}", f"{R['accuracy']:.4f}"],
                ["Precision", "TP / (TP + FP)", f"{T['precision']:.4f}", f"{R['precision']:.4f}"],
                ["Recall", "TP / (TP + FN)", f"{T['recall']:.4f}", f"{R['recall']:.4f}"],
                ["F1-score", "2 × P × R / (P + R)", f"{T['f1']:.4f}", f"{R['f1']:.4f}"],
                ["Specificity", "TN / (TN + FP)", f"{T['specificity']:.4f}", f"{R['specificity']:.4f}"],
                ["ROC AUC", "area under the ROC curve", f"{T['auc']:.4f}", f"{R['auc']:.4f}"]],
          widths=[1.6, 2.2, 1.2, 1.2], size=10)
    n = fig(doc, F + "aat2_roc.png", n, "ROC curves on the held-out test set and the real-world set", 4.2)
    n = fig(doc, F + "aat2_predictions.png", n, "Sample predictions on real eye photographs", 6.3)
    n = fig(doc, F + "aat2_gradcam.png", n, "Grad-CAM heatmaps showing where the CNN looks", 5.4)
    para(doc, "System-level test on real driver videos", size=12.5, align="left", space_after=3)
    para(doc, f"The complete pipeline (eye extraction, CNN and frame counting) was also run on {SYS['frames']:,} frames of "
              f"real driver videos from the Driver Drowsiness Dataset, which the eye CNN never saw. For each person the "
              f"share of frames with closed eyes (PERCLOS) was measured in the drowsy clip and the alert clip. The drowsy "
              f"clip had the higher PERCLOS for {SYS['drowsy_clip_higher']} of {SYS['people_with_both_clips']} people, "
              f"with an average of {pct(SYS['mean_perclos_drowsy'])} against {pct(SYS['mean_perclos_alert'])}.")
    n = fig(doc, F + "aat2_perclos.png", n, "Share of closed-eye frames per person in drowsy and alert videos", 6.4)
    para(doc, "Application screenshots", size=12.5, align="left", space_after=3)
    para(doc, "The same driver was uploaded to the Gradio application once with open eyes and once with closed eyes. "
              "The face box turns green for an alert driver and red for a drowsy driver, and the two eye regions are "
              "marked in yellow.")
    figure_pair(doc, F + "aat2_app_alert_panel.png", F + "aat2_app_drowsy_panel.png",
                f"Fig {n}: Gradio application output for an alert driver (left) and a drowsy driver (right)", width=3.05)
    n += 1

    para(doc, "Discussion of the results", size=12.5, align="left", space_after=3)
    bullets(doc, [
        f"**Held-out performance:** {pct(T['accuracy'])} accuracy and an AUC of {T['auc']:.3f} on {T['n']:,} eyes from faces not used in training.",
        f"**Real-world generalisation:** on {R['n']:,} real photographs from a different source, accuracy is {pct(R['accuracy'])} with an AUC of {R['auc']:.3f}. The drop shows the domain gap between the training images and real cameras, and is the more realistic figure.",
        f"**Missed closures:** recall on real eyes is {R['recall']:.4f}, so {R['fn']:,} of {R['tp'] + R['fn']:,} closed eyes were missed. Because the alarm needs 15 closed frames, a single missed frame only delays the alert.",
        f"**False alarms:** {R['fp']:,} of {R['tn'] + R['fp']:,} open real eyes were called Closed. Requiring 15 consecutive frames filters out most of these isolated errors.",
        "**What the CNN looks at:** Grad-CAM highlights the eyelid and the iris, which confirms that the decision is based on the eye itself and not on the background.",
        f"**Drowsiness signal:** the drowsy videos had more closed-eye frames for {SYS['drowsy_clip_higher']} of {SYS['people_with_both_clips']} people, which shows that eye closure is a useful drowsiness signal even though some drowsy clips mostly show open eyes."])
    para(doc, "The training images are AI-generated faces, and the real-world set is small and contains noisy, rotated "
              "images, so these figures show the level of performance and should not be read as a safety certification.")

    heading(doc, "5. Software Used")
    bullets(doc, ["**Python 3.9:** the main programming language.",
                  "**TensorFlow 2.20 / Keras:** used to build, train and evaluate the CNN, including augmentation layers.",
                  "**OpenCV 4:** face detection with the Haar cascade, eye cropping and the real-time webcam loop.",
                  "**NumPy:** numerical operations and storage of the prepared datasets.",
                  "**Pillow and PyArrow:** decoding the images stored in the Hugging Face Parquet files.",
                  "**Scikit-learn:** confusion matrix, ROC curve, AUC and the person-wise evaluation.",
                  "**Matplotlib:** training curves, confusion matrices, Grad-CAM and all diagrams.",
                  "**Gradio:** the web interface for uploading a driver image and viewing the result.",
                  "**Hugging Face Datasets:** source of the eye-state, real eye and driver drowsiness data."])

    heading(doc, "6. Conclusion")
    para(doc, "DriveAlert is a CNN-based system that detects driver drowsiness from a camera by checking, in every frame, "
              "whether the eyes are open or closed, and by raising an alarm when they stay closed for 15 frames in a row.")
    para(doc, f"The eye-state CNN reached {pct(T['accuracy'])} accuracy on held-out eyes and {pct(R['accuracy'])} on real "
              f"photographs from a different source. A whole-face classifier was tried first and rejected, because it "
              f"reached only {pct(FACE['accuracy'])} on unseen people. Grad-CAM showed that the network focuses on the "
              f"eyelids and iris, and the full pipeline found more closed-eye frames in the drowsy videos for "
              f"{SYS['drowsy_clip_higher']} of {SYS['people_with_both_clips']} people.")
    para(doc, "Future improvements could include training on more real driver images, including infrared images for night "
              "driving, adding yawning and head nodding as further signs of fatigue, using a facial landmark detector for "
              "more precise eye localisation, and testing the system in a real vehicle before any practical use.")
    return n
