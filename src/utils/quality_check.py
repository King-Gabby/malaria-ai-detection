"""
Quality Check — Slide/Image Quality Assessment
Classical CV-based quality checks for medical images.
"""

import cv2
import numpy as np
from typing import Dict, List, Any


def assess_image_quality(image_bgr: np.ndarray, modality: str = "microscopy") -> Dict[str, Any]:
    """
    Assess medical image quality before inference.

    Args:
        image_bgr: Input image in BGR format
        modality: 'microscopy', 'mri', 'ct', 'xray'

    Returns:
        Dict with quality metrics and pass/fail status
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    issues = []

    # 1. Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if modality == "microscopy":
        # Microscopy: high frequency detail critical
        if laplacian_var < 50:
            issues.append({
                'type': 'blur',
                'severity': 'error',
                'message': 'Image is severely blurred. Please refocus the microscope.',
                'detail': f'Sharpness: {laplacian_var:.1f} (min: 50)',
            })
        elif laplacian_var < 100:
            issues.append({
                'type': 'blur',
                'severity': 'warning',
                'message': 'Image slightly out of focus. Results may be less reliable.',
                'detail': f'Sharpness: {laplacian_var:.1f} (recommended: >100)',
            })
    elif modality in ["mri", "ct"]:
        # MRI/CT: moderate sharpness acceptable
        if laplacian_var < 20:
            issues.append({
                'type': 'blur',
                'severity': 'error',
                'message': 'Image is severely blurred.',
                'detail': f'Sharpness: {laplacian_var:.1f} (min: 20)',
            })
        elif laplacian_var < 50:
            issues.append({
                'type': 'blur',
                'severity': 'warning',
                'message': 'Image may be slightly blurred.',
                'detail': f'Sharpness: {laplacian_var:.1f} (recommended: >50)',
            })
    else:  # xray
        if laplacian_var < 30:
            issues.append({
                'type': 'blur',
                'severity': 'error',
                'message': 'X-ray is severely blurred. Retake recommended.',
                'detail': f'Sharpness: {laplacian_var:.1f} (min: 30)',
            })
        elif laplacian_var < 80:
            issues.append({
                'type': 'blur',
                'severity': 'warning',
                'message': 'X-ray slightly unsharp.',
                'detail': f'Sharpness: {laplacian_var:.1f} (recommended: >80)',
            })

    # 2. Brightness/Exposure
    mean_brightness = float(gray.mean())

    if modality == "microscopy":
        if mean_brightness < 50:
            issues.append({
                'type': 'dark',
                'severity': 'error',
                'message': 'Image too dark. Check microscope illumination.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (min: 50)',
            })
        elif mean_brightness < 80:
            issues.append({
                'type': 'dark',
                'severity': 'warning',
                'message': 'Image darker than optimal. Consider increasing illumination.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (recommended: >80)',
            })
        elif mean_brightness > 220:
            issues.append({
                'type': 'bright',
                'severity': 'error',
                'message': 'Image overexposed. Reduce illumination intensity.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (max: 220)',
            })
        elif mean_brightness > 200:
            issues.append({
                'type': 'bright',
                'severity': 'warning',
                'message': 'Image may be slightly overexposed.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (recommended: <200)',
            })
    elif modality in ["mri", "ct"]:
        # MRI/CT: wider acceptable range
        if mean_brightness < 20:
            issues.append({
                'type': 'dark',
                'severity': 'warning',
                'message': 'Image appears dark. Check window/level settings.',
                'detail': f'Mean intensity: {mean_brightness:.1f}/255',
            })
        elif mean_brightness > 230:
            issues.append({
                'type': 'bright',
                'severity': 'warning',
                'message': 'Image appears bright. Check window/level settings.',
                'detail': f'Mean intensity: {mean_brightness:.1f}/255',
            })
    else:  # xray
        if mean_brightness < 40:
            issues.append({
                'type': 'dark',
                'severity': 'error',
                'message': 'X-ray underexposed. Increase kVp/mAs.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (min: 40)',
            })
        elif mean_brightness > 210:
            issues.append({
                'type': 'bright',
                'severity': 'error',
                'message': 'X-ray overexposed. Decrease kVp/mAs.',
                'detail': f'Brightness: {mean_brightness:.1f}/255 (max: 210)',
            })

    # 3. Saturation/Clipping
    saturated_pct = np.sum(image_bgr >= 255) / image_bgr.size * 100

    if modality == "microscopy":
        sat_thresh_error = 10
        sat_thresh_warn = 5
    elif modality in ["mri", "ct"]:
        sat_thresh_error = 5
        sat_thresh_warn = 2
    else:  # xray
        sat_thresh_error = 8
        sat_thresh_warn = 3

    if saturated_pct > sat_thresh_error:
        issues.append({
            'type': 'bright',
            'severity': 'error',
            'message': f'Severe pixel saturation ({saturated_pct:.1f}%). Detail lost.',
            'detail': f'Saturated pixels: {saturated_pct:.1f}% (max: {sat_thresh_error}%)',
        })
    elif saturated_pct > sat_thresh_warn:
        issues.append({
            'type': 'bright',
            'severity': 'warning',
            'message': f'Some pixel saturation ({saturated_pct:.1f}%).',
            'detail': f'Saturated pixels: {saturated_pct:.1f}% (recommended: <{sat_thresh_warn}%)',
        })

    # 4. Contrast (standard deviation)
    contrast = float(gray.std())

    if modality == "microscopy":
        if contrast < 20:
            issues.append({
                'type': 'contrast',
                'severity': 'warning',
                'message': 'Low contrast. Staining or illumination may be suboptimal.',
                'detail': f'Contrast (std): {contrast:.1f} (recommended: >20)',
            })
    elif modality == "xray":
        if contrast < 30:
            issues.append({
                'type': 'contrast',
                'severity': 'warning',
                'message': 'Low contrast on X-ray. Check technique factors.',
                'detail': f'Contrast (std): {contrast:.1f} (recommended: >30)',
            })

    # 5. Resolution check
    min_dim = min(h, w)
    if modality == "microscopy" and min_dim < 400:
        issues.append({
            'type': 'resolution',
            'severity': 'warning',
            'message': f'Low resolution ({w}x{h}). Minimum 400px recommended.',
            'detail': f'Dimensions: {w}x{h}',
        })
    elif modality in ["mri", "ct", "xray"] and min_dim < 256:
        issues.append({
            'type': 'resolution',
            'severity': 'warning',
            'message': f'Low resolution ({w}x{h}). Minimum 256px recommended.',
            'detail': f'Dimensions: {w}x{h}',
        })

    has_errors = any(i['severity'] == 'error' for i in issues)

    return {
        'issues': issues,
        'metrics': {
            'sharpness': round(laplacian_var, 2),
            'brightness': round(mean_brightness, 1),
            'contrast': round(contrast, 1),
            'saturation_pct': round(float(saturated_pct), 1),
            'dimensions': f"{w}x{h}",
        },
        'passed': not has_errors,
        'has_warnings': len(issues) > 0,
    }


def assess_dicom_quality(dicom_dataset) -> Dict[str, Any]:
    """Assess DICOM image quality from pydicom dataset."""
    # Extract pixel data
    pixel_array = dicom_dataset.pixel_array

    # Normalize to 0-255
    if pixel_array.dtype != np.uint8:
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to BGR if grayscale
    if len(pixel_array.shape) == 2:
        image_bgr = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = pixel_array

    # Determine modality
    modality = getattr(dicom_dataset, 'Modality', '').lower()
    if modality == 'mr':
        mod = 'mri'
    elif modality == 'ct':
        mod = 'ct'
    elif modality in ['cr', 'dx', 'rf']:
        mod = 'xray'
    else:
        mod = 'unknown'

    return assess_image_quality(image_bgr, mod)