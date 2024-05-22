import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import libs.autoencoder
import libs.clip


def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'../../data/coco/coco256_features'
    latent = clip.encode(prompts)
    print(latent.shape)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()
