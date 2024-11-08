import cv2
import numpy as np



# Step 1: Correct Rolling Shift --> here I have assumed a horizontal rolling shift
# Step 2: Correct Color Shift Using Known Points


def flip_image_halves(img):
    # Ottieni le dimensioni dell'immagine
    height, width, _ = img.shape
    
    # Dividi l'immagine in due metà
    top_half = img[:height//2, :]  # La metà superiore
    bottom_half = img[height//2:, :]  # La metà inferiore
    
    # Fai il flip delle metà
    flipped_img = np.vstack((bottom_half, top_half))  # Unisce la metà inferiore con la metà superiore
    
    # Restituisce l'immagine corretta
    return flipped_img


def normalize_image(image):
    """Normalizza l'immagine da un intervallo [50, 200] a [0, 255]."""
    # Converte l'immagine in un tipo di dato a virgola mobile per evitare overflow o underflow
    normalized_img = (image.astype(np.float32) - 50) / (200 - 50) * 255
    
    # Limita i valori tra 0 e 255
    normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)
    
    return normalized_img


def get_pixel_bgr_value(image, x, y):
    """Restituisce il valore BGR di un pixel specifico dato dalle coordinate (Y, X)."""
    # Prendi il valore BGR del pixel
    b, g, r = image[x, y]

    bgr_value = (b, g, r)
    
    return bgr_value


def get_pixel_rgb_value(image, x, y):
    """Restituisce il valore RGB di un pixel specifico dato dalle coordinate (Y, X)."""
    # Prendi il valore BGR del pixel
    b, g, r = image[x, y]
    
    # Converte da BGR a RGB
    rgb_value = (r, g, b)
    
    return rgb_value


def split_image_vertically(image):
    """Divide un'immagine a metà lungo l'altezza e restituisce le due sottoimmagini."""
    # Ottieni l'altezza e la larghezza dell'immagine
    height, width, _ = image.shape
    
    # Dividi l'immagine in due metà
    top_half = image[:height // 2, :]  # Prima metà (superiore)
    bottom_half = image[height // 2:, :]  # Seconda metà (inferiore)
    
    return top_half, bottom_half



def correct_image_bgr(image, y, x, target_bgr):
    """
    Corrects the color of an entire image based on the difference between the actual
    BGR value of a given pixel and a target BGR value, using OpenCV standards.
    
    Parameters:
        image (numpy.ndarray): The image to correct.
        y (int): The y-coordinate of the reference pixel.
        x (int): The x-coordinate of the reference pixel.
        target_bgr (tuple): The target BGR value that the pixel at (y, x) should match.
    
    Returns:
        corrected_image (numpy.ndarray): The color-corrected image.
    """
    # Get the actual BGR value of the pixel at (y, x)
    actual_bgr = image[y, x].astype(np.int16)  # Convert to int16 for safe subtraction
    
    # Calculate the difference between the actual BGR and the target BGR
    bgr_difference = np.array(target_bgr, dtype=np.int16) - actual_bgr
    
    # Create an image with the same shape as the original, filled with the BGR difference
    adjustment = np.ones_like(image, dtype=np.int16) * bgr_difference
    
    # Apply the adjustment
    corrected_image = image.astype(np.int16) + adjustment
    
    # Clip values to the range [0, 255] and convert back to uint8
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    
    return corrected_image



# Load the corrupted image
image_path = "corrupted.png"  # Replace with your image file path

# Carica l'immagine
img = cv2.imread(image_path)

# Stampa le dimensioni dell'immagine
print("Dimensioni dell'immagine:", img.shape)

flipped_img = flip_image_halves(img)

# Normalizza l'immagine
normalized_image = normalize_image(flipped_img)



# Step 2: Correct Color Shift Using Known Points

# Dividi l'immagine
top_half, bottom_half = split_image_vertically(normalized_image)

# Coordinates and target BGR value
y1, x1 = 267, 564  
y2, x2 = 541-480, 128  

target_bgr_1 = (40, 195, 240) 
target_bgr_2 = (250, 158, 3)  

# Apply the correction
corrected_upper_img = correct_image_bgr(top_half, y1, x1, target_bgr_1) 
corrected_bottom_img = correct_image_bgr(bottom_half, y2, x2, target_bgr_2)

# Combine the corrected images
corrected_image = np.vstack((corrected_upper_img, corrected_bottom_img))


# Visualizza l'immagine corretta
cv2.imshow("Flipped Image",corrected_image)
# Save image 
cv2.imwrite("flipped_image.png", corrected_image)
# Aspetta che l'utente premi un tasto per chiudere la finestra
cv2.waitKey(0)
cv2.destroyAllWindows()