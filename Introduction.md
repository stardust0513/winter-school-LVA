# Windows Setup: Anaconda + VS Code + Creating a Python Environment

This guide is **Windows-only** and covers installing **Anaconda**, installing **Visual Studio Code**, and creating a **Conda Python environment**.

---

## 1) Install Anaconda

1. Download Anaconda (official site):  
   https://www.anaconda.com/download

2. Run the installer and follow the steps:
   - Installation type: **Just Me** (recommended)
   - Destination folder: keep default unless you have a specific preference

---

## 2) Install Visual Studio Code

1. Download VS Code:  
   https://code.visualstudio.com/

2. Run the installer.

3. Recommended options during installation:
   - **Add to PATH** (lets you run `code` from terminal)
   - **Open with Code** context menu (optional but convenient)

---

## 3) Install the Python Extension in VS Code

1. Open **VS Code**
2. Open **Extensions** (`Ctrl + Shift + X`)![alt text](images/image.png)
3. Install python related package. ![alt text](images/image-1.png)
4. Install Jupyter related package![alt text](images/image-2.png)
  
---

## 4) Verify Anaconda Installation

1. Open **Anaconda Prompt** (Start Menu → search **Anaconda Prompt**)![alt text](images/image-3.png)
2. Run:
```bash
conda --version
python --version
```
---

## 5) Create Python Environment

1. Open **Anaconda Propmpt**
2. Create the environment and activate it:
```bash
# Create a new conda environment (choose a name you like)
conda create -n myenv python=3.10 -y

# Activate the environment
conda activate myenv
```

## 6) Python Package

After activating the python environment, now we can install packages to be used

### Jupyter
```bash
pip install jupyter
```

Website: https://jupyter.org/

Documentation: https://docs.jupyter.org/en/latest/

Jupyter provides an interactive computing environment (Notebook and JupyterLab) that runs in the browser. It allows you to write and execute code, visualize results, and document experiments with Markdown, making it ideal for data analysis, research, teaching, and rapid prototyping.

### Pandas
```bash
pip install pandas
```

Website: https://pandas.pydata.org/

Documentation: https://pandas.pydata.org/docs/

Pandas is a powerful library for data manipulation and analysis. It offers high-level data structures such as DataFrame and Series for cleaning, transforming, aggregating (groupby), and working with time series data, as well as convenient I/O tools for CSV, Excel, SQL, and more.

### Numpy
```bash
pip install numpy
```

Website: https://numpy.org/

Documentation: https://numpy.org/doc/

NumPy is the foundational package for scientific computing in Python. It provides fast N-dimensional arrays (ndarray) and vectorized operations, along with functionality for linear algebra, broadcasting, random number generation, and efficient numerical computation.

### Scipy
```bash
pip install scipy
```

Website: https://scipy.org/

Documentation: https://docs.scipy.org/doc/scipy/

SciPy builds on NumPy and provides a broad set of scientific computing tools, including optimization, signal processing, statistics, interpolation, sparse matrices, and numerical integration. It is widely used in engineering, research, and advanced data analysis workflows.

### Pytorch
For cuda version.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For CPU version
```bash
pip install torch torchvision
```

It's suggested to install CPU version first, for the GPU version(cuda), some drivers need to be installed first. Please check the official website to have a look at the details.

Website: https://pytorch.org/

Installation Guide: https://pytorch.org/get-started/locally/

Documentation: https://pytorch.org/docs/stable/index.html

PyTorch is a widely used deep learning framework featuring dynamic computation graphs and automatic differentiation (autograd). It supports GPU acceleration for training neural networks and includes a rich ecosystem.

* torch: core tensor computation and neural network building/training
* torchvision: computer vision datasets, pretrained models, and image transforms

### Matplotlib
```bash
pip install matplotlib
```

Website: https://matplotlib.org/

Documentation: https://matplotlib.org/stable/

Matplotlib is a comprehensive plotting library for creating static, animated, and interactive visualizations in Python. It supports common scientific charts such as line plots, scatter plots, bar charts, and histograms, and integrates well with NumPy and Pandas for exploratory data analysis and reporting.