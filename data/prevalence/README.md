# Prevalence of image formats and padding

`images.csv.gz` lists 1,576,676 unique images from a 2023 web crawl of the 100K top-ranked websites (Dornauer and Felderer, 2023), one image per row in random order.
It was generated after the camera-ready version of the paper, to collect additional details (color, quantization table, and quality).
URLs, websites, and file sizes are omitted, so that the rows cannot be traced back to the images.

| Column | Description |
|---|---|
| `format` | image format reported by the crawl |
| `width`, `height` | image size reported by the crawl, if any |
| `status` | for JPEGs: `decoded`, `download failed` (in 2025), `not a jpeg`, `decoding failed`, or `not processed` |
| `color` | `color` or `gray`, for decoded JPEGs |
| `qt`, `qf` | source and quality of an exactly matching quantization table, for decoded JPEGs; `standard` is the libjpeg table |

The aggregates match the paper closely, but not exactly, see `experiments/README.md`.
