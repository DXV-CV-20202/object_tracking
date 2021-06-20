# Requirements
Install the required Python modules in file `requirements.txt`:
```
    pip3 install -r requirements.txt
```

# Scripts
## Run development server (auto reload on code change)

```
    export FLASK_ENV=development
    python3 app.py
```


## Run with custom frame count, address and port
```
    python3 app.py [args]
```

Arguments:
* ```-i, --ip```: Defaults to ```localhost```
* ```-o, --port```: Defaults to ```5000```
* ```-f, --frame-count```: Defaults to ```32```
