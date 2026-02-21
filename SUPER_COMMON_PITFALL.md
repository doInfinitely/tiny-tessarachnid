There is a retina that is 1024x1024. You take any input and scale it to fit in the retina preserving aspect ratio, going either end to end left and right, or top and bottom. Then you pad to fill the retina. But this is the key part of the algorithm: you recurse on a detected region, applying any detections to the retina via a scale, pad, and rotation (for the v4 model).

The pipeline is:

1. Crop the detected region's bbox from the parent
2. Rotate in-place to straighten (same canvas size, no expand, no re-crop)
3. scale_and_pad into the 1024x1024 retina

No canvas expansion, no tight re-crop of the rotated contour.
