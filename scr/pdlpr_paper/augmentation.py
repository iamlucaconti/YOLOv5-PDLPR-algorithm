import numpy as np
import random
import cv2
from PIL import Image, ImageFilter, ImageDraw
from torchvision import transforms

class RandomColorPad:
    def __init__(self, pad=(10, 20, 10, 20)):
        self.pad = pad

    def __call__(self, img):
        color = tuple(random.randint(0, 255) for _ in range(3))
        return transforms.functional.pad(img, padding=self.pad, fill=color)

class RandomMotionBlur:
    def __init__(self, p=0.5, kernel_size=(3, 9)):
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

class AddNoise:
    """Apply random Gaussian noise to an image with a given probability."""
    
    def __init__(self, p=0.5, noise_level=(0.02, 0.08)):
        """
        Parameters
        -----------
        - p: float
            Probability of applying the noise. Default is 0.5.
        - noise_level: tuple
            Range for standard deviation of Gaussian noise.
        """
        self.p = p
        self.noise_level = noise_level

    def __call__(self, img):
        if random.random() > self.p:
            return img
        np_img = np.array(img).astype(np.float32) / 255.0
        noise_std = random.uniform(self.noise_level[0], self.noise_level[1])
        noise = np.random.normal(0, noise_std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        return Image.fromarray((np_img * 255).astype(np.uint8))


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
    def __init__(self, p=0.5, intensity=(0.4, 0.8)):
        """
        p: probabilità di applicare l'effetto
        intensity: intervallo per il fattore di scurimento/contrasto
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
