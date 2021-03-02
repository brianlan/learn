# learn

## Generate documents

Install Sphinx and theme.


```
pip install -U sphinx
pip install catalyst-sphinx-theme
```

Run below commands to auto generate documents (rst files) for each python module and then build html. (Assume you are running in Linux)

```
cd doc
sphinx-apidoc -o ./source ..
make html
```

Start simple web server to validate.

```
cd build/html
python -m http.server
```
