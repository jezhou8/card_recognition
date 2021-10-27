# Card Scanner

Card Scanner is a WIP utility used to scan and classify a single playing card. There are currently two supported versions, one using PICamera, one with USB camera.
This is currently a WIP which means there are many bugs.

TO USE ON PI CAMERA, POSITION OF CARDS NEED TO BE FIXED

## Libraries Needed

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, opencv-python, picamera.

```bash
pip install <package name>
```

## Usage For PI Camera

For PI camera, first generate config
```bash
python configurator.py
```

Generate Training Images Using CardScannerPi
```bash
python CardScannerPi.py
```

Classify new cards using
```bash
python classify_pi.py
```


## Contributing
Not open to pull requests ATM. 

## License
[MIT](https://choosealicense.com/licenses/mit/)