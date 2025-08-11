import os
import mlflow
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch


def log_artifacts(
    image_name,
    masks,
    scores,
    height,
    width,
    image,
    target,
    channels
):
    with tempfile.TemporaryDirectory() as tmpdir:
        if channels == 4:
            image, dom = image[..., :3], image[..., 3]
        for i, (score, mask) in enumerate(zip(scores, masks)):
            if channels == 4:
                fig, axs = plt.subplots(1, 4, figsize=(12, 3))
                axs[0].imshow(image)
                axs[1].imshow(dom, cmap="gray")
                axs[2].imshow(target, cmap="gray")
                axs[3].imshow(mask, cmap="gray")

                for ax in axs:
                    ax.axis("off")

                axs[0].set_title("Input RGB")
                axs[1].set_title("Input DOM")
                axs[2].set_title("Target mask")
                axs[3].set_title(f"Predicted mask, score: {score:.3f}")
            else:
                fig, axs = plt.subplots(1, 3, figsize=(12, 3))
                axs[0].imshow(image)
                axs[1].imshow(target, cmap="gray")
                axs[2].imshow(mask, cmap="gray")

                for ax in axs:
                    ax.axis("off")

                axs[0].set_title("Input RGB")
                axs[1].set_title("Target mask")
                axs[2].set_title(f"Predicted mask, score: {score:.3f}")

            plt.tight_layout()
            overlay_path = os.path.join(tmpdir, f"{image_name}_mask{i}.png")
            fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    
        # Log the entire folder as an MLflow artifact
        mlflow.log_artifacts(tmpdir, artifact_path=f"predicted_masks/{image_name}")


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    dataset,
    channels,
    height=512,
    width=512,
):

    # Train mask decoder
    predictor.model.sam_mask_decoder.train(True)

    # Train prompt encoder
    predictor.model.sam_prompt_encoder.train(True)

    predictor.model.eval()

    avg_score = 0.0
    for idx in range(dataset.__len__()):
        image, target = dataset[idx]
        predictor.set_image(image)
        masks, scores, _ = predictor.predict()
        avg_score += max(scores)
        output_mask_name = f"{dataset.__getfilename__(idx)}"
        log_artifacts(
            output_mask_name,
            masks,
            scores,
            height,
            width,
            image,
            target,
            channels
        )
    avg_score /= dataset.__len__()
    mlflow.log_metric("avg_iou_score", avg_score)