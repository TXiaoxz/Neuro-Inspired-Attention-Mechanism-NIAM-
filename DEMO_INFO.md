# NIAM v2 Refined Demo

## Status
Demo is currently running at: **http://localhost:7860**

## Access the Demo

### Local Access
Open your browser and navigate to:
```
http://localhost:7860
```

### Network Access
If you want to access from other devices on the same network:
```
http://YOUR_SERVER_IP:7860
```

## Features

- Professional clean interface
- Real-time audio enhancement
- Upload or record audio directly
- Pre-loaded example samples
- Model performance metrics displayed

## Model Information

- **Architecture**: Transformer with NIAM v2 Refinement
- **Training Data**: 50,000 speech samples with cocktail party noise
- **Performance**: 5.6% WER improvement (53.3% to 50.3%)
- **Validation Loss**: -17.5026 dB SI-SNR

## Usage

1. Upload a noisy audio file or record from microphone
2. Click "Enhance Audio" button
3. Listen to the enhanced output
4. Try the example samples provided

## Stopping the Demo

To stop the demo server:
```bash
pkill -f demo_app.py
```

## Restarting the Demo

To restart the demo:
```bash
cd /home/dlwlx05/project/NIAM/Neuro-Inspired-Attention-Mechanism-NIAM--main
conda activate niam
python demo_app.py
```

## Background Mode

To run in background with logs:
```bash
nohup python demo_app.py > demo.log 2>&1 &
```

View logs:
```bash
tail -f demo.log
```
