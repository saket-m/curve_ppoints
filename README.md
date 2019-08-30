# curve_ppoints
Semantic segmentation mask >> get individual masks >> get curve control points for the outline of the mask

# To get control points from a RGB mask having all the classes
python get_bezier.py "RGB semantic segmentation mask" "output folder" "algorithm" "slope thresold if 'algorithm' is 1" "1 if upsample vertices"
<br>
e.g. python get_bezier.py test1/roto-2.png masks/roto-lr-vup-2/ 1 25 1

# To get control points from a grey scale mask having onlly one individua image
python gen_spline.py "grey scale individual mask path"  "mask num" "slope thresold"
e.g. python gen_spline.py masks/anand/mask200.png  200 25
