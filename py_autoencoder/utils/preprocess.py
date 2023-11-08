from scipy.spatial import Delaunay
import cv2
import numpy as np
import mediapipe as mp
import skimage
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import Delaunay

from .lm_data import FACE_OVAL

def get_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    # Convert the image to 32-bit float
    image_float32 = image.astype(np.float32)
    # Normalize the image to range [0,1]
    image_float32 = (image_float32 - np.min(image_float32)) / (np.max(image_float32) - np.min(image_float32))
    # Convert the normalized float32 image to uint8 with values in range [0, 255]
    image_uint8 = (image_float32 * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                landmark = face_landmarks.landmark[i]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    return np.array(landmarks)

def warp_image(source, target, landmarks1, landmarks2):
    # Compute Delaunay Triangulation
    delaunay = Delaunay(landmarks1)
    warped_image = np.zeros_like(source)
    
    # Iterate through each triangle in the triangulation
    for simplex in delaunay.simplices:
        # Get the vertices of the triangle in both images
        src_triangle = landmarks1[simplex]
        dest_triangle = landmarks2[simplex]

        # Compute the bounding box of the triangle in both images
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        # Check if the bounding box has non-positive width or height, and skip if it does
        if src_rect[2] <= 0 or src_rect[3] <= 0 or dest_rect[2] <= 0 or dest_rect[3] <= 0:
            # Optionally log the problematic triangle information here
            print(f"Skipping triangle with src_rect: {src_rect} or dest_rect: {dest_rect}")
            continue

        # Crop the triangle from the source and destination images
        src_cropped_triangle = source[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        dest_cropped_triangle = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)

        # Adjust coordinates to the cropped region
        src_triangle_adjusted = src_triangle - (src_rect[0], src_rect[1])
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])

        # Compute the affine transformation
        matrix = cv2.getAffineTransform(np.float32(src_triangle_adjusted), np.float32(dest_triangle_adjusted))

        # Warp the source triangle to the shape of the destination triangle
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dest_rect[2], dest_rect[3]))

        # Mask for the destination triangle
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), (1, 1, 1), 16, 0)

        dest_img_patch = warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]]

        # Resize or slice the warped_triangle and mask if necessary to fit the destination patch
        if warped_triangle.shape[:2] != dest_img_patch.shape[:2]:
            warped_triangle = cv2.resize(warped_triangle, (dest_img_patch.shape[1], dest_img_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (dest_img_patch.shape[1], dest_img_patch.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Now the shapes should match, and you can proceed with the operation
        dest_img_patch *= (1 - mask[:, :, None])
        dest_img_patch += warped_triangle * mask[:, :, None]

        # Place the modified patch back into the image
        warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = dest_img_patch


    return warped_image.astype(np.uint8)
def warp_image_single_channel(source, target, landmarks1, landmarks2):
    # Compute Delaunay Triangulation
    delaunay = Delaunay(landmarks1)
    
    num_channels = source.shape[2] if len(source.shape) > 2 else 1
    warped_image = np.zeros_like(target if num_channels == 3 else target[:,:,None])  # Adjust for 1 or 3 channels
    
    # Iterate through each triangle in the triangulation
    for simplex in delaunay.simplices:
        # Get the vertices of the triangle in both images
        src_triangle = landmarks1[simplex]
        dest_triangle = landmarks2[simplex]

        # Compute the bounding box of the triangle in both images
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        # Crop the triangle from the source and destination images
        src_cropped_triangle = source[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        dest_cropped_triangle = np.zeros((dest_rect[3], dest_rect[2], num_channels), dtype=source.dtype)

        # Adjust coordinates to the cropped region
        src_triangle_adjusted = src_triangle - (src_rect[0], src_rect[1])
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])

        # Compute the affine transformation
        matrix = cv2.getAffineTransform(np.float32(src_triangle_adjusted), np.float32(dest_triangle_adjusted))

        # Warp the source triangle to the shape of the destination triangle
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dest_rect[2], dest_rect[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Mask for the destination triangle
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), 1, 16)

        # Place the warped triangle in the destination image
        if num_channels == 3:
            # For a 3-channel image, use stacking of the mask
            mask_stack = mask[:, :, None]
            warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] *= (1 - mask_stack)
            warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] += warped_triangle * mask_stack
        else:
            # For a 1-channel image, ensure the warped_triangle is indeed single-channel
            if warped_triangle.ndim > 2:
                warped_triangle = warped_triangle[:, :, 0]  # Extract the single-channel if necessary
                    # Inside the warp_image_single_channel function, before the operation that causes the error:
    if warped_triangle.ndim > 2:
        # This will squeeze singleton dimensions from the array.
        warped_triangle = np.squeeze(warped_triangle)

        # Now perform the operation with the corrected shapes.
        if warped_triangle.shape[:2] == mask.shape:
            # Ensure the shapes are compatible for broadcasting.
            warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] *= (1 - mask)
            warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] += warped_triangle * mask
        else:
            # Raise an error or handle the mismatch in shapes appropriately.
            raise ValueError("Shape of warped_triangle and mask do not match.")

        warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] *= (1 - mask)
        warped_image[dest_rect[1]:dest_rect[1]+dest_rect[3], dest_rect[0]:dest_rect[0]+dest_rect[2]] += warped_triangle * mask

    return warped_image if num_channels == 3 else warped_image[:,:,0]

def crop_face(image):
    # If the image has more than 3 channels, take only the first three
    if image.shape[2] > 3:
        image = image[:, :, :3]
    image = image.astype(np.float32)
    landmarks = get_landmarks(image)
    
    # If no landmarks are detected, return the original image
    if len(landmarks) == 0:
        return image
    
    # Extract the coordinates corresponding to the FACE_OVAL
    face_coords = [landmarks[i] for i in [point[1] for point in FACE_OVAL]]
    
    # Create a mask with the FACE_OVAL
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(face_coords)], (255))
    
    # Find the bounding rectangle around the face
    x, y, w, h = cv2.boundingRect(mask)
    w = h = min(w, h)
    bounding_box = np.array([x, y, h, h])
    # Crop the image based on this rectangle
    cropped_face = image[y:y+h, x:x+h]

    return cropped_face, bounding_box

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def mask_face_oval(image):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # If no landmarks are found, return the original image
    if not results.multi_face_landmarks:
        return image

    # Create a mask of zeros with the same shape as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Get the landmarks for the face. The landmarks creating an oval around the face typically include the jawline, forehead, and cheek landmarks.
    face_oval_landmarks = []
    for landmark in results.multi_face_landmarks[0].landmark:
        # The landmark.x and landmark.y values are normalized [0.0, 1.0] values
        # Convert them to image coordinates.
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        face_oval_landmarks.append((x, y))

    # Considering that the FACE_OVAL landmarks include the complete boundary of the face, create a convex hull.
    hull = cv2.convexHull(np.array(face_oval_landmarks))
    cv2.drawContours(mask, [hull], 0, (255), -1)

    # Use the mask to get the region of the original image that lies inside the FACE_OVAL
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    return masked_img


def extract_face_skin_area(img):
    landmarks_points = get_landmarks(img)
    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask based on FACE_OVAL landmarks
    face_oval_mask = np.zeros_like(hsv_img[:, :, 0])
    face_oval_indices = [point[1] for point in FACE_OVAL]
    face_oval_points = np.array([landmarks_points[idx] for idx in face_oval_indices], dtype=np.int32)
    cv2.fillPoly(face_oval_mask, [face_oval_points], 1)

    # Use the mask to extract the face oval area
    masked_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=face_oval_mask)

    return masked_hsv
def threshold_face_skin_area(img):

    masked_hsv = extract_face_skin_area(img)

    # Compute mean and standard deviation for each channel using the masked area
    mean_hue = np.mean(masked_hsv[:,:,0][masked_hsv[:,:,0] > 0])
    std_hue = np.std(masked_hsv[:,:,0][masked_hsv[:,:,0] > 0])
    mean_sat = np.mean(masked_hsv[:,:,1][masked_hsv[:,:,1] > 0])
    std_sat = np.std(masked_hsv[:,:,1][masked_hsv[:,:,1] > 0])
    mean_val = np.mean(masked_hsv[:,:,2][masked_hsv[:,:,2] > 0])
    std_val = np.std(masked_hsv[:,:,2][masked_hsv[:,:,2] > 0])

    # Define thresholds based on the mean and standard deviation for each channel
    lower_bound = [max(0, mean_hue - 1.5*std_hue), max(0, mean_sat - 1.5*std_sat), max(0, mean_val - 1.5*std_val)]
    upper_bound = [max(180, mean_hue + 1.5*std_hue), min(255, mean_sat + 1.5*std_sat), min(255, mean_val + 1.5*std_val)]

    # Convert lists to numpy arrays
    LOWER_THRESHOLD = np.array(lower_bound, dtype=np.uint8)
    UPPER_THRESHOLD = np.array(upper_bound, dtype=np.uint8)

    # Create a binary mask where the skin color is within the threshold
    skinMask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), LOWER_THRESHOLD, UPPER_THRESHOLD)

    # Extract skin regions using the mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    return skin


from scipy.constants import h, c, k
def remove_lighting_from_hsi(hsi_data, temperature=3000):
    """Remove the lighting influence from a hyperspectral image using Planck's blackbody radiation."""
    h = 6.626e-34  # Planck's constant (in m^2 kg / s)
    c = 3.0e8      # Speed of light (in m/s)
    k = 1.38e-23   # Boltzmann's constant (in m^2 kg s^-2 K^-1)
    # Limit to the first 31 channels of HSI data
    hsi_data = hsi_data[:,:,:31]
    # Define the relevant wavelengths in meters
    wavelengths = np.arange(400e-9, 790e-9, 10e-9)[:31]
    """Calculate the spectral radiance of a blackbody using Planck's law."""
    planck = (8 * np.pi * h * c) / (wavelengths**5 * (np.exp((h * c) / (wavelengths * k * temperature)) - 1))
    # Compute the blackbody radiation and normalize
    blackbody_contrib = planck
    blackbody_normalized = blackbody_contrib / np.linalg.norm(blackbody_contrib)
    # Determine the scaling factors
    scaling_factors = np.einsum('ijk,k->ij', hsi_data, blackbody_normalized)
    # Scale the blackbody contribution for each pixel
    scaled_blackbody = scaling_factors[:, :, np.newaxis] * blackbody_normalized
    # Subtract the scaled blackbody contribution from the original data
    # hsi_data_corrected = hsi_data + (0.1 * scaled_blackbody)
    hsi_data_corrected = hsi_data
    return hsi_data_corrected

def get_light_spectra():
    h = 6.626e-34  # Planck's constant (in m^2 kg / s)
    c = 3.0e8      # Speed of light (in m/s)
    k = 1.38e-23   # Boltzmann's constant (in m^2 kg s^-2 K^-1)
    temperature = 2850
    wavelengths = np.arange(400e-9, 790e-9, 10e-9)[:39]
    """Calculate the spectral radiance of a blackbody using Planck's law."""
    planck = (8 * np.pi * h * c) / (wavelengths**5 * (np.exp((h * c) / (wavelengths * k * temperature)) - 1))
    # Compute the blackbody radiation and normalize
    blackbody_contrib = planck
    blackbody_normalized = blackbody_contrib / np.linalg.norm(blackbody_contrib)
    print(blackbody_normalized)
    return blackbody_normalized

def add_thickness_as_alpha(rgb_image_path):
    # Hardcoded paths for thickness map and model image
    thickness_map_path = r"C:\Desktop\m104_thickness_map.png"
    model_path = r"C:\Users\joeli\Dropbox\Data\models_4k\m53_4k.png"
    
    # Read and process the thickness map
    thickness = cv2.imread(thickness_map_path, cv2.IMREAD_GRAYSCALE)
    thickness = cv2.resize(thickness, (1024, 1024))
    
    # Read and process the model image
    model = cv2.imread(model_path)
    model = cv2.cvtColor(model, cv2.COLOR_BGR2RGB)
    model = cv2.resize(model, (1024, 1024))
    
    # Read and process the RGB image
    rgb_data = cv2.imread(rgb_image_path)
    rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
    rgb_data = cv2.resize(rgb_data, (1024, 1024))
    
    # Get landmarks
    landmarks_model = get_landmarks(model)
    landmarks_rgb = get_landmarks(rgb_data)
    
    # Warp the model image to match the RGB image
    warped_model = warp_image(model, rgb_data, landmarks_model, landmarks_rgb)
    warped_model, bounding_box = crop_face(warped_model)
    
    # Prepare the thickness map
    thickness_3_channel = np.stack((thickness,)*3, axis=-1)
    warped_thickness_3_channel = warp_image(thickness_3_channel, rgb_data, landmarks_model, landmarks_rgb)
    warped_thickness = warped_thickness_3_channel[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], 0]
    
    # Normalize the thickness map and add as alpha channel
    warped_thickness = (warped_thickness - np.min(warped_thickness)) / (np.max(warped_thickness) - np.min(warped_thickness))
    thickness_alpha = (warped_thickness * 0.2 + 0.01) * 255
    thickness_alpha = thickness_alpha.astype(np.uint8)
    
    # Crop the RGB image to match the warped model
    rgb_cropped = rgb_data[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    
    # Combine the RGB image with the thickness map as alpha channel
    rgba_image = np.dstack((rgb_cropped, thickness_alpha))

    # Save the RGBA image
    cv2.imwrite("rgb_thickness_alpha.png", cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))
    
    return rgba_image, warped_thickness