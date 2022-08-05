"""
Simple utility to render an .svg to a .png
"""
import os
import argparse
import pydiffvg
import torch as th
from tqdm import tqdm
from CLIPasso.chunk_processor import is_complete

def render(canvas_width, canvas_height, shapes, shape_groups):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    return img


def main(args):
    pydiffvg.set_device(th.device('cuda:1'))

    sketches_dir = args.sketches_dir # Batch file, each line is an image path

    sketch_chunk_folders = os.listdir(sketches_dir)
    # Each chunk folder contains folders named after some image whose sketches were created with CLIPasso
    completed_images = {}
    for sketch_chunk_folder in sketch_chunk_folders:
        images_folders = os.listdir(os.path.join(sketches_dir, sketch_chunk_folder))
        images_folders.sort()
        chunk_dir_txt = os.path.join(args.chunks_dir, sketch_chunk_folder + ".txt")
        with open(chunk_dir_txt, "r") as f:
            image_paths = f.read().splitlines()
            image_paths.sort()

        for i, images_folder in enumerate(tqdm(images_folders)):
            full_path = os.path.join(sketches_dir, sketch_chunk_folder, images_folder)
            if is_complete(full_path):
                root, folders, _ = next(os.walk(full_path))
                for folder in folders:
                    
                    sketch = os.path.join(root, folder, "best_iter.svg")
                    sketch_out = os.path.join(root, folder, "best_iter2.png")
                    if os.path.exists(sketch_out):
                        continue

                    # Load SVG
                    svg = os.path.join(sketch)
                    canvas_width, canvas_height, shapes, shape_groups = \
                        pydiffvg.svg_to_scene(svg)

                    # Save initial state
                    ref = render(canvas_width, canvas_height, shapes, shape_groups)
                    pydiffvg.imwrite(ref.cpu(), sketch_out, gamma=2.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sketches_dir", help="source clipasso batch file path")
    parser.add_argument("chunks_dir", help="output image path")
    args = parser.parse_args()
    sketches_dir = "./data/sketch/clipasso/Horses/clipsketches"
    chunks_dir = "./data/sketch/clipasso/Horses/chunkinator/chunks"
    main(args)
