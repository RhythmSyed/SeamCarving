Usage:
SeamCarving(input_filename, output_filename, dimension_change, energy_algorithm, display_seams)

input_filename: path to the image desired
output_filename: specify a filepath for the processed output image
dimension_change: provide an aspect ratio change for the input image. This is a delta value, for example '0, 100' means 0 pixels added to height, 100 pixels added to width
energy_algorithm: 'backward' or 'forward'
display_seams: Boolean True of False to display seams as an output or not


Example execution:
python main.py 'images/fig5.png' 'fig5-seam_removal-backward.png' '0, -120' 'backward' False

Commands for generated Images:
    Figure 5 from the 2007 paper:
        python main.py 'images/fig5.png' 'fig5-seam_removal-backward.png' '0, -350' 'backward' True
        python main.py 'images/fig5.png' 'fig5-seam_removal-forward.png' '0, -350' 'forward' True

    Figure 8 from the 2007 paper -- parts c, d, and f only
        python main.py 'images/fig8.png' 'fig8-seam_insertion-backward.png' '0, 100' 'backward' True
        python main.py 'images/fig8.png' 'fig8-seam_insertion-forward.png' '0, 100' 'forward' True
        python main.py 'images/fig8.png' 'fig8-seam_insertion-backward.png' '0, 200' 'backward' True
        python main.py 'images/fig8.png' 'fig8-seam_insertion-forward.png' '0, 200' 'forward' True

    Figure 8 (bench) from the 2008 paper
        python main.py 'images/fig8-2008.png' 'fig8-2008-seam_removal-backward.png' '0, -100' 'backward' True
        python main.py 'images/fig8-2008.png' 'fig8-2008-seam_removal-forward.png' '0, 100' 'forward' True

     Figure 9 (elongated car) from the 2008 paper
        python main.py 'images/fig9-2008.png' 'fig9-2008-seam_insertion-backward.png' '0, 100' 'backward' True
        python main.py 'images/fig9-2008.png' 'fig9-2008-seam_insertion-forward.png' '0, 100' 'forward' True
        python main.py 'images/fig9-2008.png' 'fig9-2008-seam_removal-backward.png' '0, -100' 'backward' True
        python main.py 'images/fig9-2008.png' 'fig9-2008-seam_removal-forward.png' '0, -100' 'forward' True