import os
import glob
import numpy as np
import cv2
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from skimage.filters import gaussian, laplace


class MRIEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing metrics on {self.device}...")

        # Data range is 0.0 - 1.0 (we normalize images to this range)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @staticmethod
    def compute_hfen(pred, target, sigma=1.5):
        """
        Computes High Frequency Error Norm (HFEN).
        Critical for MRI to measure how well edges and fine details are reconstructed.
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.squeeze().cpu().numpy()

        # Apply LoG (Laplacian of Gaussian) filter
        # Sigma 1.5 is the standard setting for MRI HFEN
        lo_g_pred = laplace(gaussian(pred, sigma=sigma))
        lo_g_target = laplace(gaussian(target, sigma=sigma))

        # HFEN = ||LoG(pred) - LoG(target)||_2 / ||LoG(target)||_2
        numerator = np.linalg.norm(lo_g_pred - lo_g_target)
        denominator = np.linalg.norm(lo_g_target)

        return numerator / (denominator + 1e-8)

    @staticmethod
    def compute_nmse(pred, target):
        """
        Computes Normalized Mean Squared Error (NMSE).
        Standard error metric for medical image reconstruction.
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.squeeze().cpu().numpy()

        mse = np.linalg.norm(pred - target) ** 2
        norm_gt = np.linalg.norm(target) ** 2
        return mse / (norm_gt + 1e-8)

    @staticmethod
    def eval_all_metrics(
        ground_truth: torch.Tensor,
        generated: torch.Tensor,
        psnr: PeakSignalNoiseRatio,
        ssim: StructuralSimilarityIndexMeasure,
    ) -> tuple[float]:
        return (
            MRIEvaluator.compute_hfen(generated, ground_truth),
            MRIEvaluator.compute_nmse(generated, ground_truth),
            psnr(generated, ground_truth).item(),
            ssim(generated, ground_truth).item(),
        )

    def evaluate_folders(self, generated_dir, ground_truth_dir):
        exts = ["*.png", "*.jpg", "*.JPG"]  # , '*.jpeg', '*.tif', '*.tiff']

        gen_files = sorted(
            [f for ext in exts for f in glob.glob(os.path.join(generated_dir, ext))]
        )
        gt_files = sorted(
            [f for ext in exts for f in glob.glob(os.path.join(ground_truth_dir, ext))]
        )

        if len(gen_files) != len(gt_files):
            print(
                f"Warning: File count mismatch. Gen: {len(gen_files)}, GT: {len(gt_files)}"
            )

        # Removed LPIPS from tracking
        metrics_sum = {"PSNR": 0.0, "SSIM": 0.0, "HFEN": 0.0, "NMSE": 0.0}
        count = 0

        print(f"Starting evaluation of {len(gen_files)} pairs...")

        for gen_path, gt_path in zip(gen_files, gt_files):
            # 1. Load Images (Grayscale)
            img_gen = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if img_gen is None or img_gt is None:
                print(f"Error reading pair: {gen_path}")
                continue

            # 2. Preprocess: Normalize to [0, 1]
            img_gen = img_gen.astype(np.float32) / 255.0
            img_gt = img_gt.astype(np.float32) / 255.0

            # Convert to Tensor (N, C, H, W) -> (1, 1, H, W)
            t_gen = torch.from_numpy(img_gen).unsqueeze(0).unsqueeze(0).to(self.device)
            t_gt = torch.from_numpy(img_gt).unsqueeze(0).unsqueeze(0).to(self.device)

            # 3. Compute Metrics
            try:
                metrics_sum["PSNR"] += self.psnr(t_gen, t_gt).item()
                metrics_sum["SSIM"] += self.ssim(t_gen, t_gt).item()
                metrics_sum["HFEN"] += MRIEvaluator.compute_hfen(img_gen, img_gt)
                metrics_sum["NMSE"] += MRIEvaluator.compute_nmse(img_gen, img_gt)
                count += 1
            except Exception as e:
                print(f"Error computing metrics for {os.path.basename(gen_path)}: {e}")

            if count % 10 == 0:
                print(f"Processed {count} images...")

        # 4. Final Report
        if count == 0:
            print("No images processed.")
            return

        results = {k: v / count for k, v in metrics_sum.items()}

        print("\n" + "=" * 40)
        print(" FINAL MRI SR METRICS ")
        print("=" * 40)
        print(f" Images Processed: {count}")
        print("-" * 40)
        print(f" PSNR  (↑) : {results['PSNR']:.4f} dB")
        print(f" SSIM  (↑) : {results['SSIM']:.4f}")
        print(f" NMSE  (↓) : {results['NMSE']:.4f}")
        print(f" HFEN  (↓) : {results['HFEN']:.4f} (Edge Fidelity)")
        print("=" * 40)

        return results


if __name__ == "__main__":
    GEN_FOLDER = "GEN_FOLDER"
    GT_FOLDER = "GT_FOLDER"

    evaluator = MRIEvaluator()
    evaluator.evaluate_folders(GEN_FOLDER, GT_FOLDER)
