import numpy as np
import random
import cv2
from PIL import Image, ImageFilter, ImageDraw
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch


class RandomColorPad:
    """
     Apply random padding to an image with a specified color.

    Padding is applied symmetrically to the top/bottom and left/right
    with randomly selected values within given ranges.

    """
    def __init__(self, pad_y_range=(5, 15), pad_x_range=(10, 30), color_pad = "black"):
        """
        Parameters
        ----------
        pad_y_range : tuple of int
            Range (min_y, max_y) for vertical padding (top and bottom). Default is (5, 15).
        pad_x_range : tuple of int
            Range (min_x, max_x) for horizontal padding (left and right). Default is (10, 30).
        color_pad : str
            Padding color mode. Use "black" for black padding (0), or any other string
            to sample a random RGB color. Default is "black".

        Notes
        -----
        - The input image must be a PIL Image.
        - The padding is applied using torchvision.transforms.functional.pad.
        
        """
        
        self.pad_y_range = pad_y_range  # (min_y, max_y)
        self.pad_x_range = pad_x_range  # (min_x, max_x)
        self.color_pad = color_pad

    def __call__(self, img):
        # Padding casuale per asse Y (top e bottom)
        vertical_pad = random.randint(*self.pad_y_range)

        # Padding casuale per asse X (left e right)
        horizontal_pad = random.randint(*self.pad_x_range)

        # Colore casuale
        if self.color_pad == "black":
            color = 0
        else:
            color = tuple(random.randint(0, 255) for _ in range(3))

        return TF.pad(img, (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad), fill=color)
    
    
class RandomMotionBlur:
    """
    Apply a motion blur effect to a PIL image with a given probability.

    The blur simulates camera motion by convolving the image with a directional linear kernel.

    """
    def __init__(self, p=0.5, kernel_size=(3, 9)):
        """
        Parameters
        ----------
        p : float
            Probability of applying the motion blur. Default is 0.5.
        kernel_size : tuple
            Range of kernel sizes to sample from (min, max). Only odd integers are used.

        Notes
        -----
        - The image is first converted to a NumPy array for OpenCV processing.
        - A random odd kernel size and blur direction (angle in degrees) are selected.
        - The kernel is applied using OpenCV's `filter2D`.
        - The blurred image is then converted back to a PIL Image.
        """
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Choose random kernel size (must be odd)
        k = random.choice(range(self.kernel_size[0], self.kernel_size[1] + 1, 2))

        # Choose a random angle (direction of jitter)
        angle = random.uniform(0, 360)
        kernel = self._motion_blur_kernel(k, angle)

        # Apply the kernel to the image (separately to each channel)
        blurred = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(blurred)

    def _motion_blur_kernel(self, kernel_size, angle):
        # Create a straight line kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        kernel[center, :] = np.ones(kernel_size)
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        # Normalize
        kernel /= kernel.sum()
        return kernel

class RandomGaussianBlur:
    """Apply Gaussian blur to an image with a given probability and variable radius."""

    def __init__(self, p=0.3, radius=(0.5, 1.5)):
        """
        Parameters
        ----------
        - p : float
            Probability of applying Gaussian blur. Default is 0.3.
        - radius : tuple
            Range of blur radius (min, max). A random value is sampled from this range.

        Notes
        -----
        - The input image must be a PIL Image.
        - The Gaussian blur is applied using PIL's ImageFilter.GaussianBlur.
        """
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(self.radius[0], self.radius[1])
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


class SimulateDistance:
    """
    Simulates the effect of distance by scaling down and then upsampling an image,
    resulting in a loss of detail that mimics lower resolution.
    """
    def __init__(self, scale_range=(0.4, 0.8), p=0.4):
        """
        Parameters
        ----------
        - scale_range : tuple of float
            Range of scaling factors to simulate distance (min_scale, max_scale). 
            A random scale is sampled from this range. Default is (0.4, 0.8).
        - p : float
            Probability of applying the transformation. Default is 0.4.

        Notes
        -----
        - The input image must be a PIL Image.
        - The image is first downsampled using `Image.NEAREST` interpolation, then upsampled
        back to the original size, resulting in a pixelated or blurred effect.
        """

        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        img = img.resize((new_w, new_h), resample=Image.NEAREST)

        return img.resize((w, h), resample=Image.NEAREST)


class AddFog:
    """
    Simulates a foggy effect by blending the image with a white overlay.
    """
    
    def __init__(self, p=0.3, fog_factor=(0.2, 0.6)):
        """
        Parameters
        ----------
        - p : float
            Probability of applying the fog effect. Default is 0.3.
        - fog_factor : tuple of float
            Range of blending factors (min_alpha, max_alpha). A random value is sampled
            from this range to determine the fog intensity.

        Notes
        -----
        - The input image must be a PIL Image.
        - The fog is simulated by blending the original image with a white RGB image using
        `Image.blend`.
        """
        self.p = p
        self.fog_factor = fog_factor

    def __call__(self, img):
        if random.random() > self.p:
            return img
        fog = Image.new("RGB", img.size, (255, 255, 255))
        alpha = random.uniform(*self.fog_factor)
        return Image.blend(img, fog, alpha)


class MatrixEffect:
    """
    Apply a custom green-tinted "Matrix"-style effect to a PIL image with a given probability.
    The effect darkens and alters the RGB channels to emphasize green tones.
    
    """
    def __init__(self, p=0.5, intensity=(0.4, 0.8)):
        """
        Parameters
        ----------
        p : float
            Probability of applying the Matrix effect. Default is 0.5.
        intensity : tuple of float
            Range of the contrast/darkening factor applied to the image. Values are typically between 0.4 and 0.8.

        Notes
        -----
        - The image is converted to a NumPy array and normalized to [0, 1] for processing.
        - The red and blue channels are reduced more than green to simulate a green-dominant color scheme.
        - A higher contrast factor is applied when the average luminance is low.
        - The result is converted back to a PIL Image.
        """
        self.p = p
        self.intensity = intensity

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Modifica canali per effetto "Matrix"
        r = img_np[..., 0] * 0.4
        g = img_np[..., 1] * 0.65
        b = img_np[..., 2] * 0.4

        img_matrix = np.stack([r, g, b], axis=-1)
        
        luminance = img_matrix.mean()

        # Intensità casuale di scurimento
        factor = random.uniform(*self.intensity)
        if luminance < 0.25:
            factor = max(factor, 0.8) 

        img_matrix = img_matrix * factor

        img_matrix = np.clip(img_matrix * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_matrix)

class RandomLightBeam:
    def __init__(self, 
                 intensity=(0.4, 0.8),   # intervallo intensità del fascio
                 angle_range=(-30, 30),  # angolo in gradi
                 beam_width_range=(20, 60),  # larghezza fascio in pixel
                 beam_type="white"       # "white", "black" oppure "random"
                 ):
        self.intensity = intensity
        self.angle_range = angle_range
        self.beam_width_range = beam_width_range
        self.beam_type = beam_type

    def __call__(self, img):
        img = img.convert('RGBA')
        w, h = img.size

        beam_center_x = random.randint(0, w)
        beam_width = random.randint(*self.beam_width_range)
        angle = random.uniform(*self.angle_range)

        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        left = max(beam_center_x - beam_width // 2, 0)
        right = min(beam_center_x + beam_width // 2, w)
        draw.rectangle([left, 0, right, h], fill=255)
        mask = mask.rotate(angle, expand=False, fillcolor=0)

        mask_np = np.array(mask).astype(np.float32) / 255
        intensity = random.uniform(*self.intensity)

        # Decidi colore fascio
        beam_type = self.beam_type
        if beam_type == "random":
            beam_type = random.choice(["white", "black"])

        if beam_type == "white":
            beam_color = (255, 255, 255, 0)
            alpha = (mask_np * 255 * intensity).clip(0, 255).astype(np.uint8)
        elif beam_type == "black":
            beam_color = (0, 0, 0, 0)
            alpha = (mask_np * 255 * intensity).clip(0, 255).astype(np.uint8)
        else:
            raise ValueError("beam_type must be 'white', 'black' or 'random'")

        beam_img = Image.new('RGBA', (w, h), beam_color)
        beam_img.putalpha(Image.fromarray(alpha))

        out = Image.alpha_composite(img, beam_img).convert('RGB')

        return out


class BluePlateHighlight:
    def __init__(self, intensity_range=(1.3, 1.7), p=0.3):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        # Convert to numpy array
        img_np = np.array(img).astype(np.float32) / 255.0

        # Boost the blue channel
        blue_boost = random.uniform(*self.intensity_range)
        img_np[..., 2] = np.clip(img_np[..., 2] * blue_boost, 0, 1)

        # Convert back to PIL
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)



class BlockShiftTransform:
    def __init__(self, 
                 direction='both', 
                 num_blocks_range_horizontal=(4, 12), 
                 num_blocks_range_vertical=(4, 12), 
                 max_shift=10, 
                 p=0.5):
        """
        direction: 'horizontal', 'vertical', or 'both'
        num_blocks_range_horizontal: tuple (min, max) blocchi orizzontali
        num_blocks_range_vertical: tuple (min, max) blocchi verticali
        max_shift: massimo spostamento (in pixel) per blocco
        p: probabilità di applicare la trasformazione
        """
        self.direction = direction
        self.num_blocks_range_horizontal = num_blocks_range_horizontal
        self.num_blocks_range_vertical = num_blocks_range_vertical
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img = TF.to_tensor(img)
        C, H, W = img.shape

        img_np = img.permute(1, 2, 0).numpy()  # H x W x C

        if self.direction in ['horizontal', 'both']:
            num_blocks = random.randint(*self.num_blocks_range_horizontal)
            block_height = H // num_blocks
            for i in range(num_blocks):
                y_start = i * block_height
                y_end = H if i == num_blocks - 1 else (i + 1) * block_height
                shift = random.randint(-self.max_shift, self.max_shift)
                img_np[y_start:y_end] = np.roll(img_np[y_start:y_end], shift, axis=1)

        if self.direction in ['vertical', 'both']:
            num_blocks = random.randint(*self.num_blocks_range_vertical)
            block_width = W // num_blocks
            for i in range(num_blocks):
                x_start = i * block_width
                x_end = W if i == num_blocks - 1 else (i + 1) * block_width
                shift = random.randint(-self.max_shift, self.max_shift)
                img_np[:, x_start:x_end] = np.roll(img_np[:, x_start:x_end], shift, axis=0)

        img_out = torch.tensor(img_np).permute(2, 0, 1).clamp(0, 1)

        return TF.to_pil_image(img_out)


class DitherEffect:
    """
    Apply a dithering effect to a PIL image using a specified palette mode and dithering algorithm.

    Dithering simulates color depth reduction by diffusing pixels, which can give a stylized or retro appearance.

    """
    def __init__(self, dither=Image.FLOYDSTEINBERG, palette_mode='P', p=1.0):
        """
        Parameters
        ----------
        dither : int
            Dithering method used during conversion. Default is `Image.FLOYDSTEINBERG`.
            Other options include `Image.NONE`.
        palette_mode : str
            Color mode for dithering. Default is 'P' (8-bit palette mode).
        p : float
            Probability of applying the effect. Default is 1.0 (always applied).

        Notes
        -----
        - The image is first converted to the specified palette mode using the given dithering method.
        - Then it's converted back to RGB to keep compatibility with further processing steps.
        - This transformation can simulate a pixelated or vintage look depending on the parameters.
        """
        self.dither = dither
        self.palette_mode = palette_mode
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.convert(self.palette_mode, dither=self.dither)
            img = img.convert('RGB')
        return img


# -----------------------------------------------------------------------------------------------
## Simulate DB
LowResolutionTransform = transforms.Compose([
    transforms.Resize((24, 72), interpolation=transforms.InterpolationMode.BILINEAR),  # downscale
    transforms.Resize((48, 144), interpolation=transforms.InterpolationMode.BILINEAR)  # upscale
])

transform_night = transforms.Compose([

    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.9, 1.1),
            degrees=(-8, 8),              # to simulate "rotate", "tilt". "challenge"
            translate=(0.0, 0.10),
            shear=(-15, 15, -5, -5),                
            fill=0
        )], p=0.6),
    
    transforms.RandomPerspective(distortion_scale=0.20, p=0.50),
     
    transforms.RandomApply([
    RandomLightBeam(intensity=(0.1, 0.3), angle_range=(-45, 45), beam_width_range=(10, 60), beam_type = "black")],
                            p=0.6),

    transforms.RandomApply([
    RandomLightBeam(intensity=(0.1, 0.3), angle_range=(-45, 45), beam_width_range=(10, 60), beam_type = "black")],
                            p=0.6),
    
    
    MatrixEffect(intensity=(0.5, 0.6), p=0.9),
    BluePlateHighlight(intensity_range=(1, 2), p=0.9),
    
    
    # DitherEffect(dither=Image.NONE, palette_mode='P', p=0.9),
    
    RandomMotionBlur(kernel_size=(5, 7), p=0.3),
    RandomGaussianBlur(radius=(0.5, 0.8), p=0.8),
    
    #   transforms.RandomApply([
    #     LowResolutionTransform
    # ], p=0.5),
    
    transforms.RandomApply([
        transforms.ColorJitter(  
            brightness=(0.4, 0.7),     
            contrast=(1, 2.5),       
            )], p=0.70),
    
    
    transforms.RandomApply([
        transforms.ColorJitter(       
            saturation=(0.5, 1),     
            )], p=0.60),
   
    transforms.RandomApply([
        transforms.ColorJitter(   
            brightness=(0.5, 0.9),     
            contrast=(1, 2),       
            )], p=0.70),

    transforms.RandomApply([
        transforms.ColorJitter(     
            # contrast=(1, 2),       
            saturation=(0.4, 1),     
            )], p=0.20),

    SimulateDistance(scale_range=(0.6, 0.9), p=0.4), 
    
    transforms.ToTensor()
])


# Simulate brightness
transform_day = transforms.Compose([
        transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.9, 1.1),
            degrees=(-10, 10),              # to simulate "rotate", "tilt". "challenge"
            translate=(0.0, 0.10),
            shear=(-15, 15, -10, -10),                
            fill=0
        )
    ], p=0.4),
    
    transforms.RandomPerspective(distortion_scale=0.20, p=0.20),
    
    transforms.RandomApply([
        transforms.ColorJitter(   # For "db" and "challenge" datasets
            brightness=(1.2, 2),     
            contrast=(1, 1.5),       
            saturation=(0.6, 1.4),     
            hue=(-0.05, 0.05)),
    ], p=0.8),
    
     DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.7),
     RandomMotionBlur(kernel_size=(5, 7), p=0.30),
    
    transforms.RandomApply([
        LowResolutionTransform
    ], p=0.6),
    
    transforms.RandomApply([
    RandomLightBeam(intensity=(0.5, 0.9), angle_range=(-60, 60), beam_width_range=(20, 80), beam_type = "white")],
                            p=0.4),
    
    
    AddFog(fog_factor=(0.2, 0.7), p=0.6),
    
    SimulateDistance(scale_range=(0.6, 0.9), p=0.8), 
    
    transforms.CenterCrop((48, 144)),
    transforms.ToTensor()
])

## Simulate FN
# To handle CPD-FN: The distance from the LP to the shooting location is relatively far or near.
LowResolutionTransform = transforms.Compose([
    transforms.Resize((24, 72), interpolation=transforms.InterpolationMode.BILINEAR),  # downscale
    transforms.Resize((48, 144), interpolation=transforms.InterpolationMode.BILINEAR)  # upscale
])

transform_fn = transforms.Compose([

    transforms.RandomApply([
        transforms.ColorJitter(   # For "db" and "challenge" datasets
             brightness=(0.6, 1.0),     
            contrast=(0.8, 1.2),       
            saturation=(0.4, .8),     
            hue=(-0.05, 0.05)),
    ], p=0.8),

    transforms.RandomApply([
        transforms.RandomAffine(
            # scale=(0.9, 1.1),
            degrees=(-5, 5),              # to simulate "rotate", "tilt". "challenge"
            # translate=(0.10, 0.10),
            shear=(-10, 10, -10, 10),                
            fill=0
        )
    ], p=0.3),
    
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.4),
    RandomMotionBlur(kernel_size=(7, 9), p=0.6),
    BlockShiftTransform(
        direction='both',
        num_blocks_range_horizontal=(8, 16),
        num_blocks_range_vertical=(10, 20), 
        max_shift=1, 
        p=0.2
    ),
    
    transforms.Resize((48, 144)),
    transforms.RandomApply([
        LowResolutionTransform
    ], p=0.70),
    
    SimulateDistance(scale_range=(0.25, 0.45), p=0.4),
    transforms.ToTensor()
])


## Simulate Blur
LowResolutionBlur = transforms.Compose([
    transforms.Resize((15, 45), interpolation=transforms.InterpolationMode.BILINEAR),  # downscale
    transforms.Resize((48, 144), interpolation=transforms.InterpolationMode.NEAREST)])  # upscale
                                            
# To handle with CCPD-Blur
transform_blur = transforms.Compose([

    BluePlateHighlight(intensity_range=(1, 1.6), p=0.6),
    
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=(0.7, 1.3),     
            contrast=(0.7, 1.3),       
            saturation=(0.7, 1.3),     
            hue=(-0.1, 0.1)),
    ], p=0.70),
    
    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.9, 1),
            degrees=(-10, 10),              # to simulate "rotate", "tilt". "challenge"
            translate=(0.0, 0.10),
            shear=(-10, 10, -10, 10),                
            fill=0
        )
    ], p=0.5),
    
    DitherEffect(dither=Image.NONE, palette_mode='P', p=0.7),
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.7),
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.7),
    
    RandomMotionBlur(kernel_size=(5, 7), p=0.9),
    
    transforms.RandomApply([
        LowResolutionBlur
    ], p=0.90),
    
    RandomGaussianBlur(radius=(0.4, 0.9), p=0.9),
    transforms.ToTensor()
])


## Simulate rot
# CCPD-Rotate Great horizontal tilt degree (20◦ - 50◦) and the vertical tilt degree varies from -10◦ to 10◦.
transform_rot = transforms.Compose([
    transforms.RandomApply([RandomColorPad(pad_y_range=(55, 60), pad_x_range=(30, 35), color_pad='random')], p=0.99),
    
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=(0.6, 1.4),     
            contrast=(0.6, 1.4),       
            saturation=(0.6, 1.4),     
            hue=(-0.08, 0.08)),
    ], p=0.90),
    
    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.8, 1),
            degrees=(-20, 20),              # to simulate "rotate", "tilt". "challenge"
            translate=(0.0, 0.0),
            shear=(-10, 10, -10, 10),                
            fill=0
        )
    ], p=0.95),
    
   transforms.RandomPerspective(distortion_scale=0.3, p=0.30),
    
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.4),
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.4),
    
    RandomMotionBlur(kernel_size=(5, 7), p=0.4),
    SimulateDistance(scale_range=(0.7, 1), p=0.30),

    
    transforms.CenterCrop((70, 150)),
    transforms.Resize((48, 144)),
    transforms.CenterCrop((44, 120)),
    transforms.Resize((48, 144)),
    transforms.ToTensor()
])


## Simulate Tilt
# To handle with CCPD-Tilt Great horizontal tilt degree and vertical tilt degree.
transform_tilt_1 = transforms.Compose([
    transforms.RandomApply([RandomColorPad(pad_y_range=(55, 60), pad_x_range=(55, 60), color_pad='random')], p=1),
    
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=(0.8, 1.2),     
            contrast=(0.8, 1.2),       
            saturation=(0.8, 1.2),     
            hue=(-0.1, 0.1)),
    ], p=0.7),

    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.8, .9),
            degrees=(-10, 10),              
            shear=(-40, 40, -20, -15),                
            fill=0
        )
    ], p=0.99),
    
    transforms.RandomPerspective(distortion_scale=0.4, p=0.40),
    
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.4),
    transforms.CenterCrop((70, 120)),
    transforms.Resize((48, 144)),
    
    MatrixEffect(intensity=(0.8, 0.1), p=0.0),
    RandomMotionBlur(kernel_size=(6, 8), p=0.2),
    SimulateDistance(scale_range=(0.7, 1), p=0.20),
   
    transforms.ToTensor()
])

transform_tilt_2 = transforms.Compose([
    transforms.RandomApply([RandomColorPad(pad_y_range=(55, 60), pad_x_range=(55, 60), color_pad='random')], p=1),
    
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=(0.8, 1.2),     
            contrast=(0.8, 1.2),       
            saturation=(0.8, 1.2),     
            hue=(-0.1, 0.1)),
    ], p=0.7),

    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.8, .9),
            degrees=(-10, 10),              
            shear=(-40, 40, 15, 20),                
            fill=0
        )
    ], p=0.99),
    
    transforms.RandomPerspective(distortion_scale=0.4, p=0.40),
    
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.4),
    transforms.CenterCrop((70, 120)),
    transforms.Resize((48, 144)),
    
    MatrixEffect(intensity=(0.8, 0.1), p=0.0),
    RandomMotionBlur(kernel_size=(6, 8), p=0.2),
    SimulateDistance(scale_range=(0.7, 1), p=0.20),
   
    transforms.ToTensor()
])

## Simulate Challenge
LowResolutionChallenge = transforms.Compose([
    transforms.Resize((15, 45), interpolation=transforms.InterpolationMode.BILINEAR),  # downscale
    transforms.Resize((48, 144), interpolation=transforms.InterpolationMode.NEAREST)])  # upscale
                                            
# To handle with CCPD-Challenge: The most challenging images for LPDR to date.
transform_challenge = transforms.Compose([
     transforms.RandomApply([
    RandomLightBeam(intensity=(0.2, 0.4), angle_range=(-60, 60), beam_width_range=(10, 70), beam_type = "black")],
                            p=0.4),

    transforms.RandomApply([
    RandomLightBeam(intensity=(0.2, 0.4), angle_range=(-60, 60), beam_width_range=(10, 70), beam_type = "white")],
                            p=0.4),

    BluePlateHighlight(intensity_range=(1, 1.6), p=0.6),
    
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=(0.7, 1.3),     
            contrast=(0.7, 1.3),       
            saturation=(0.7, 1.3),     
            hue=(-0.1, 0.1)),
    ], p=0.70),
    
    transforms.RandomApply([
        transforms.RandomAffine(
            scale=(0.9, 1),
            degrees=(-10, 10),              # to simulate "rotate", "tilt". "challenge"
            translate=(0.0, 0.10),
            shear=(-10, 10, -10, 10),                
            fill=0
        )
    ], p=0.5),
    
     DitherEffect(dither=Image.NONE, palette_mode='P', p=0.7),
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.7),
    DitherEffect(dither=Image.FLOYDSTEINBERG, palette_mode='P', p=0.7),
    
    BlockShiftTransform(
        direction='both',
        num_blocks_range_horizontal=(4, 8),
        num_blocks_range_vertical=(8, 16), 
        max_shift=1, 
        p=0.6
    ),
    RandomMotionBlur(kernel_size=(5, 7), p=0.9),
    transforms.RandomApply([
        LowResolutionChallenge
    ], p=0.90),
    
    RandomGaussianBlur(radius=(0.4, 0.9), p=0.9),
    transforms.ToTensor()
])

# Only normalization (no augmentation) for the validation set
val_transform = transforms.Compose([
    transforms.Resize((48, 144)),
    transforms.ToTensor(),
])