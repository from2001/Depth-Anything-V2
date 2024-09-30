import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import concurrent.futures  # for multi-threading

# Set the environment variable to enable MPS fallback
# CPU will be used if CUDA is not available.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from depth_anything_v2.dpt import DepthAnythingV2

def process_file(depth_anything, filename, args, cmap):
    print(f'Processing: {filename}')
    
    raw_image = cv2.imread(filename)
    
    if raw_image is None:
        print(f"Failed to read image: {filename}")
        return
    
    depth = depth_anything.infer_image(raw_image, args.input_size)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
    
    if args.pred_only:
        saved = cv2.imwrite(output_path, depth)
        if saved:
            print(f"Image saved at: {output_path}")
        else:
            print(f"Failed to save image at: {output_path}")
    else:
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth])
        
        saved = cv2.imwrite(output_path, combined_result)
        if saved:
            print(f"Image saved at: {output_path}")
        else:
            print(f"Failed to save image at: {output_path}")
    
    # Clear memory after processing
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Execute file processing in parallel using multi-threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, depth_anything, filename, args, cmap) for filename in filenames]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during processing: {e}")
