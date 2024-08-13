from matplotlib import pyplot as plt


class ImagePlotter:
    @staticmethod
    def plot_images(original, modified):
        # Assuming the images are in the format [channels, height, width]
        original = original.detach().numpy().transpose(1, 2, 0)
        modified = modified.detach().numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(modified)
        ax[1].set_title("Modified by Attention")
        ax[1].axis('off')

        plt.show()