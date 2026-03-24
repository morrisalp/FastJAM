# Packages
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# Project
from .homography_utils import get_homography_matrix, apply_homography

from transformers.reflection_transformer import ReflectionTransformer
from transformers.sequence_transformer import SequenceTransformer
from transformers.homography_transformer import HomographyTransformer

from .device_utils import get_device
device = get_device()

def plot_warped_grid_images_canonical_single(image_paths, model, graph_data, image_size,
                                             stn_n, best_reflections, dpi=150, save_path=None):
    import os
    import matplotlib.pyplot as plt

    device = graph_data.x.device
    B = len(image_paths)
    W, H = image_size

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # === Step 1: Compute transformations ===
    model.eval()
    with torch.no_grad():
        thetas = model.forward_n(graph_data, stn_n)

    avg_theta = thetas.mean(dim=0)  # shape: (3, 3)

    # === Step 2: Load images as tensors ===
    transform = transforms.Compose([transforms.ToTensor()])
    images_tensor = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(device)

    # Use actual image dimensions (may differ from RoMa resolution)
    _, _, img_H, img_W = images_tensor.shape
    actual_size = (img_H, img_W)

    # === Step 3: Warp and plot each image ===
    for i in range(B):
        img_tensor = images_tensor[i].unsqueeze(0)  # shape: (1, 3, H, W)

        transformer = SequenceTransformer(
            [
                ReflectionTransformer(actual_size, best_reflections[i].unsqueeze(0)),
                HomographyTransformer(actual_size, torch.linalg.inv(thetas[i]).unsqueeze(0), lie_algebra=False),
                HomographyTransformer(actual_size, avg_theta.unsqueeze(0), lie_algebra=False),
            ],
            combine_transformations=True
        )

        warped_img = transformer(img_tensor)
        warped_np = (warped_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        plt.imshow(warped_np)
        plt.axis("off")

        if save_path is not None:
            fig.savefig(os.path.join(save_path, f"img_{i}_to_canonical.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

def plot_alignment_overview(image_paths, model, graph_data, image_size,
                            stn_n, best_reflections, save_path=None, max_cols=10):
    """
    Two-row overview figure: top row = original images, bottom row = aligned to canonical.
    Saved as a single PNG at save_path (directory or full file path).
    """
    device = graph_data.x.device
    B = len(image_paths)

    model.eval()
    with torch.no_grad():
        thetas = model.forward_n(graph_data, stn_n)

    avg_theta = thetas.mean(dim=0)

    transform = transforms.Compose([transforms.ToTensor()])
    images_tensor = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(device)

    _, _, img_H, img_W = images_tensor.shape
    actual_size = (img_H, img_W)

    # Warp all images
    warped = []
    for i in range(B):
        img_tensor = images_tensor[i].unsqueeze(0)
        transformer = SequenceTransformer(
            [
                ReflectionTransformer(actual_size, best_reflections[i].unsqueeze(0)),
                HomographyTransformer(actual_size, torch.linalg.inv(thetas[i]).unsqueeze(0), lie_algebra=False),
                HomographyTransformer(actual_size, avg_theta.unsqueeze(0), lie_algebra=False),
            ],
            combine_transformations=True
        )
        w = transformer(img_tensor)
        warped.append((w.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    originals = [(images_tensor[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                 for i in range(B)]

    n_cols = min(B, max_cols)
    n_rows_per_block = (B + n_cols - 1) // n_cols  # rows needed per block
    cell = img_W / 100  # figure inches per cell

    fig, axes = plt.subplots(
        2 * n_rows_per_block, n_cols,
        figsize=(n_cols * cell, 2 * n_rows_per_block * cell * (img_H / img_W)),
        squeeze=False
    )

    for idx in range(B):
        row_block = idx // n_cols
        col = idx % n_cols
        axes[row_block][col].imshow(originals[idx])
        axes[row_block][col].axis("off")
        axes[n_rows_per_block + row_block][col].imshow(warped[idx])
        axes[n_rows_per_block + row_block][col].axis("off")

    # Hide unused cells
    for idx in range(B, n_rows_per_block * n_cols):
        row_block = idx // n_cols
        col = idx % n_cols
        axes[row_block][col].axis("off")
        axes[n_rows_per_block + row_block][col].axis("off")

    # Row labels
    axes[0][0].set_title("Before", fontsize=8, loc="left", pad=2)
    axes[n_rows_per_block][0].set_title("After", fontsize=8, loc="left", pad=2)

    plt.tight_layout(pad=0.3)

    if save_path is not None:
        if os.path.splitext(save_path)[1] in (".png", ".jpg", ".pdf"):
            out_file = save_path
        else:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, "alignment_overview.png")
        fig.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved alignment overview: {out_file}")
    else:
        plt.show()


def plot_warped_grid_images_ref_single(image_paths, model, graph_data, image_size, stn_n, ref,
                                       save_path=None, best_reflections=None, dpi=150):
    import os
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    device = graph_data.x.device
    B = len(image_paths)
    W, H = image_size

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Step 1: Compute transformations
    model.eval()
    with torch.no_grad():
        thetas = model.forward_n(graph_data, stn_n)

    # Step 2: Load images as tensors
    transform = transforms.Compose([transforms.ToTensor()])
    images_tensor = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(device)

    # Use actual image dimensions (may differ from RoMa resolution)
    _, _, img_H, img_W = images_tensor.shape
    actual_size = (img_H, img_W)

    # Step 3: Warp and plot each image
    for i in range(B):
        img_tensor = images_tensor[i].unsqueeze(0)  # shape: (1, 3, H, W)

        transformer = SequenceTransformer(
            [
                ReflectionTransformer(actual_size, best_reflections[i].unsqueeze(0)),
                HomographyTransformer(actual_size, torch.linalg.inv(thetas[i]).unsqueeze(0), lie_algebra=False),
                HomographyTransformer(actual_size, thetas[ref].unsqueeze(0), lie_algebra=False),
                ReflectionTransformer(actual_size, best_reflections[ref].unsqueeze(0)),
            ],
            combine_transformations=True
        )

        warped_img = transformer(img_tensor)
        warped_np = (warped_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        plt.imshow(warped_np)
        plt.axis("off")

        if save_path is not None:
            fig.savefig(os.path.join(save_path, f"img_{i}_to_{ref}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()