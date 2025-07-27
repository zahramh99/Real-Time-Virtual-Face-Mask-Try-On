# Real-Time Virtual Face Mask Try-On
> A complete solution for real-time face mask detection and virtual try-on using deep learning

## Features
- Real-time face detection with SSD MobileNet
- Mask classification (Mask/No Mask)
- Virtual try-on for multiple mask styles
- Web interface with camera controls
- Snapshot functionality
- Custom mask uploads

## Quick Start

```bash
# Clone repository
git clone https://github.com/zahramh99/face-mask-tryon.git
cd face-mask-tryon

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run application
python app.py