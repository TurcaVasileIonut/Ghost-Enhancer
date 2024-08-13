import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError


class Metrics:
    @staticmethod
    def psnr(pictureA, pictureB):
        psnr = PeakSignalNoiseRatio()
        return psnr(pictureA, pictureB)

    @staticmethod
    def ssim(pictureA, pictureB):
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(pictureA, pictureB)

    @staticmethod
    def mae(pictureA, pictureB):
        mae = MeanAbsoluteError()
        return mae(pictureA, pictureB)


if __name__ == '__main__':
    psnr = PeakSignalNoiseRatio()
    image = torch.randn(256, 256, 3)
    compressed = torch.randn(256, 256, 3)
    psnr = psnr(image, compressed)
    print(psnr)
