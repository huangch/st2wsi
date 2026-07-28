# ST2WSI_Registration

**Spatial Transcriptomics to Whole Slide Image Registration**

An ImageJ/Fiji plugin for registering DAPI images from spatial transcriptomics platforms (e.g., 10x Xenium, Visium) to H&E-stained whole slide images (WSI). This enables the integration of morphological features from H&E histopathology with spatially resolved gene expression data.

## Overview

ST2WSI_Registration implements the image registration workflow described in:

> Huang, C.-H. et al. *WSInsight: an open platform for whole slide image analytics in computational pathology.* npj Precision Oncology (2025). https://doi.org/10.1038/s41698-025-00841-9

The registration pipeline (Figure 2 in the paper) consists of:

1. **Colour deconvolution** — Extract the haematoxylin channel from H&E WSI (nuclear stain analogous to DAPI)
2. **SIFT-based affine alignment** — Coarse alignment using Scale-Invariant Feature Transform keypoints
3. **B-spline elastic registration** — Fine-grained deformable registration via bUnwarpJ

This allows spatial transcriptomics cell coordinates to be transformed onto the H&E coordinate system, enabling AI-based cell segmentation models trained on H&E (e.g., CellViT) to be applied to spatially-resolved molecular data.

## Installation

### From JAR (recommended for Fiji)

1. Download the latest release JAR from [Releases](https://github.com/huangch/st2wsi/releases)
2. Copy the JAR to your Fiji `plugins/` directory
3. Restart Fiji

### From source

```bash
git clone https://github.com/huangch/st2wsi.git
cd st2wsi
mvn clean package
cp target/ST2WSI_Registration-*.jar /path/to/Fiji.app/plugins/
```

### From Docker

Build the container image:

```bash
docker build -t st2wsi:latest .
```

Run the plugin in GUI mode (requires X11 access):

```bash
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  st2wsi:latest gui
```

Run the plugin in headless CLI mode:

```bash
docker run --rm -it \
  -v /your/data:/data \
  st2wsi:latest cli \
  -o /data/out \
  -r /data/dapi.ome.tif \
  -t /data/wsi.ome.tif \
  --ref-series 3 \
  --tgt-series 3 \
  --ref-rotated 90 \
  --tgt-channel Hematoxylon
```

### Local shell helper

A convenience wrapper is included for local headless execution:

```bash
chmod +x run_st2wsi.sh
./run_st2wsi.sh \
  -o /data/out \
  -r /data/dapi.ome.tif \
  -t /data/wsi.ome.tif \
  --ref-series 3 \
  --tgt-series 3 \
  --ref-rotated 90 \
  --tgt-channel Hematoxylon
```

## Usage

### GUI mode

1. Open Fiji
2. Go to **Plugins → ST2WSI Registration**
3. Select the output directory
4. Provide paths to:
   - **Reference image**: DAPI image from spatial transcriptomics (OME-TIFF)
   - **Target image**: H&E whole slide image (OME-TIFF, SVS, or other Bio-Formats supported format)
5. Configure series index, rotation, and colour deconvolution channel
6. Run the registration

### Headless / command-line mode

```bash
/path/to/Fiji.app/ImageJ-linux64 \
    --headless --ij2 \
    --run "st2wsi_registration.ST2WSI_Registration" \
    "outputDir=/data/out \
     refImagePath=/data/dapi.ome.tif \
     tgtImagePath=/data/wsi.ome.tif \
     refSeries=3 \
     tgtSeries=3 \
     refFlipped=false \
     refRotated=90 \
     tgtChannel=Hematoxylon"
```

The same parameters are accepted by the Docker entrypoint and the local wrapper script.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `outputDir` | Directory for output transformation files |
| `refImagePath` | Path to DAPI reference image (OME-TIFF) |
| `tgtImagePath` | Path to H&E target WSI |
| `refSeries` | Series index in the reference image (default: 0) |
| `tgtSeries` | Series index in the target image (default: 0) |
| `refFlipped` | Whether the reference is horizontally flipped (`true`/`false`) |
| `refRotated` | Rotation of reference in degrees (0, 90, 180, 270) |
| `tgtChannel` | Colour deconvolution channel: `Hematoxylon`, `Eosin`, or `DAB` |

### Output

The plugin produces:
- **Affine transformation matrix** (JSON) — for coarse alignment
- **B-spline deformation field** — for elastic registration
- **Registered images** — aligned reference overlaid on target (for QC)

## Colour Deconvolution

The plugin uses Ruifrok & Johnston's colour deconvolution algorithm to extract the haematoxylin channel from H&E images. The stain vectors are defined in `colourdeconvolution.txt`:

```
# H&E stain vectors (OD space)
Hematoxylon: 0.644 0.717 0.267
Eosin:       0.093 0.954 0.283
DAB:         0.268 0.570 0.776
```

## Dependencies

- [ImageJ](https://imagej.nih.gov/ij/) / [Fiji](https://fiji.sc/)
- [bUnwarpJ](https://imagej.net/plugins/bunwarpj) — elastic registration
- [MPICBG](https://imagej.net/plugins/feature-extraction) — SIFT feature extraction
- [Bio-Formats](https://www.openmicroscopy.org/bio-formats/) — OME-TIFF and WSI reading

## Citation

If you use ST2WSI_Registration in your research, please cite:

```bibtex
@article{huang2025wsinsight,
  title={WSInsight: an open platform for whole slide image analytics in computational pathology},
  author={Huang, Chao-Hui and others},
  journal={npj Precision Oncology},
  year={2025},
  doi={10.1038/s41698-025-00841-9}
}
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

## Acknowledgements

This plugin builds upon:
- **bUnwarpJ** by Ignacio Arganda-Carreras and Jan Kybic
- **SIFT** implementation from MPICBG by Stephan Saalfeld
