import cv2, os
from pathlib import Path
import numpy as np
from PIL import Image
import io
from IPython.display import Image as ip_im


def write_frames_to_video(
    frames,
    outpath,
    fps=24,
):
    assert outpath.endswith(".mp4")
    assert frames.ndim == 4

    outdir = Path(outpath).parent
    os.makedirs(outdir, exist_ok=True)

    # Define the video properties
    width = frames.shape[-2]
    height = frames.shape[-3]

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    # Generate some frames
    for i in range(len(frames)):
        frame = frames[i, :, :, ::-1]
        out.write(frame)

    # Release resources
    out.release()


import imageio.v3 as iio


def write_frames_ffmpeg(frames, outpath, fps=24):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = ((np.clip(frames, -1, 1) + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
    iio.imwrite(
        outpath,
        frames,
        plugin="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",  # widely compatible
        macro_block_size=None,  # allows odd sizes
    )


def frames_to_gif(frames_t, fps=24, save_path=None):
    # List of frames as numpy arrays
    frames = [
        (255 * (frames_t[:, i].clip(0, 1)).transpose(1, 2, 0)).astype(np.uint8)
        for i in range(frames_t.shape[1])
    ]

    # Convert the frames to Pillow images
    images = [Image.fromarray(frame) for frame in frames]

    # Create a BytesIO buffer
    buf = io.BytesIO()

    # Save the frames as a GIF to the buffer
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # Duration of each frame in milliseconds
        loop=0,
    )  # Loop the GIF indefinitely

    # Get the GIF data from the buffer
    gif_data = buf.getvalue()
    if save_path is not None:
        with open(save_path, "wb") as f:
            f.write(gif_data)
    return ip_im(data=gif_data)
