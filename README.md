# curve_ppoints
Semantic segmentation mask >> get individual masks >> get curve control points for the outline of the mask

# To run
python get_bezier.py <RGB semantic segmentation mask> <output folder> <algorithm> <slope thresold if <algorithm is 1>>
e.g. python get_bezier.py test1/roto-12.png masks/roto-lr-12/ 1 25

