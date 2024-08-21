# FLUX.1 DEV Inpainting Model
🎨 Fill in masked parts of images with FLUX.1-dev 🖌️

<a href='https://replicate.com/zsxkib/flux-dev-inpainting'><img src='https://replicate.com/zsxkib/flux-dev-inpainting/badge'></a>

## About
This is a version of the [Flux DEV inpainting model](https://huggingface.co/spaces/SkalskiP/FLUX.1-inpaint) by @skalskip92. We've changed it to keep the original image's shape. The model expects the mask to be the same size as the input image, but you can change this with some settings.

Big thanks to [@Gothos13](https://twitter.com/Gothos13) for helping create this clever inpainting method.

## How to Use
You can use this model with Cog. Here's an example command:

```bash
sudo cog predict \
  -i 'mask="https://replicate.delivery/pbxt/HtGQBqO9MtVbPm0G0K43nsvvjBB0E0PaWOhuNRrRBBT4ttbf/mask.png"' \
  -i 'image="https://replicate.delivery/pbxt/HtGQBfA5TrqFYZBf0UL18NTqHrzt8UiSIsAkUuMHtjvFDO6p/overture-creations-5sI6fQgYIuo.png"' \
  -i 'prompt="small cute cat sat on a park bench"' \
  -i 'strength=1' \
  -i 'num_inference_steps=30' \
  -i 'guidance_scale=7' \
  -i 'output_format="webp"' \
  -i 'output_quality=90'
```
|                                                             Input Image                                                             |                                                Mask                                                |
| :---------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| ![Input Image](https://replicate.delivery/pbxt/HtGQBfA5TrqFYZBf0UL18NTqHrzt8UiSIsAkUuMHtjvFDO6p/overture-creations-5sI6fQgYIuo.png) | ![Mask](https://replicate.delivery/pbxt/HtGQBqO9MtVbPm0G0K43nsvvjBB0E0PaWOhuNRrRBBT4ttbf/mask.png) |

**Output:**

<p align="center">
  <img src="https://pbs.twimg.com/media/GVGyNUXXkAE4oKF?format=jpg&name=medium" alt="Output Image">
</p>

## Tips for Use
- For better results, try using more steps (between 20-30) when creating the image.
- Play around with the inpainting 'strength'. Numbers between 0.85-1.0 often work well, but you might need different strengths for different prompts.
- Keep in mind that Flux wasn't specially trained for inpainting. There are no specific inpainting parts, which makes this a smart way to use the model's skills.
- The model can still handle text in images.

## Note
This inpainting method can make great images, but you might need to try a few times to get what you want. Don't give up if your first try doesn't look perfect!

For more info and updates, check out the [original tweet thread](https://x.com/zsakib_/status/1824449453249462352).

## Support
If you like my work, please follow me! [@zsakib_](https://twitter.com/zsakib_)